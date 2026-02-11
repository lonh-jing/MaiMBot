"""
ComfyUI backend (Workflow API).

This backend calls a fixed ComfyUI prompt graph that:
LoadAudio -> MLX_Qwen3TTSVoiceClone -> SaveAudioMP3

Rationale:
- ComfyUI expects API-format "prompt" graphs (not UI workflow json).
- For audio inputs, the simplest reliable path is to copy the reference audio into ComfyUI/input
  and use the built-in LoadAudio node.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
import uuid
from typing import Any, ClassVar, Dict, Optional, Tuple
from urllib.parse import urlencode

from src.common.logger import get_logger
from src.plugin_system.apis import generator_api

from .base import TTSBackendBase, TTSResult
from ..config_keys import ConfigKeys
from ..utils.file import TTSFileManager
from ..utils.session import TTSSessionManager
from ..utils.text import TTSTextUtils

logger = get_logger("tts_comfyui")


LANG_TO_DEMO = {
    "zh": "Chinese",
    "ja": "Japanese",
    "en": "English",
}


class ComfyUIBackend(TTSBackendBase):
    backend_name = "comfyui"
    backend_description = "ComfyUI 工作流 API（MLX Qwen3-TTS VoiceClone/CustomVoice）"
    support_private_chat = True
    default_audio_format = "mp3"

    _ref_cache: ClassVar[Dict[str, str]] = {}
    _instruct_cache: ClassVar[Dict[str, str]] = {}
    # If set by subclasses, only these modes are allowed (e.g. {"voice_clone"}).
    allowed_modes: ClassVar[Optional[set[str]]] = None

    def get_default_voice(self) -> str:
        return self.get_config(ConfigKeys.COMFYUI_DEFAULT_STYLE, "default")

    def _filter_styles_by_mode(self, styles: Dict[str, Any]) -> Dict[str, Any]:
        allowed = self.allowed_modes
        if not allowed:
            return styles
        out: Dict[str, Any] = {}
        for name, st in (styles or {}).items():
            if not isinstance(st, dict):
                continue
            mode = str(st.get("mode") or "voice_clone").strip()
            if mode in allowed:
                out[name] = st
        return out

    def _normalize_styles_config(self, styles_config: Any) -> Dict[str, Any]:
        # Match GPT-SoVITS backend style schema: list[{name,...}] or dict{name:{...}}
        if isinstance(styles_config, dict):
            return styles_config
        if isinstance(styles_config, list):
            result = {}
            for style in styles_config:
                if isinstance(style, dict) and "name" in style:
                    name = style["name"]
                    result[name] = {k: v for k, v in style.items() if k != "name"}
            return result
        return {}

    def _clean_instruct(self, s: str, max_chars: int) -> str:
        s = (s or "").strip()
        if not s:
            return ""

        # Strip common wrappers.
        s = s.replace("```", "").strip()
        s = re.sub(r"^instruct\\s*[:：]\\s*", "", s, flags=re.IGNORECASE).strip()

        # Prefer first non-empty line.
        for line in s.splitlines():
            line = line.strip()
            if line:
                s = line
                break

        # Trim quotes.
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1].strip()

        if max_chars and len(s) > max_chars:
            s = s[:max_chars].rstrip()
        return s

    def _clean_base_tone(self, s: str) -> str:
        """
        Clean a base tone/persona string so it can safely live inside `基调=...`:
        - single-line
        - no semicolons (they are field separators)
        - no '=' (KV separator)
        """
        s = (s or "").strip()
        if not s:
            return ""
        s = s.replace("\r", " ").replace("\n", " ")
        s = re.sub(r"\\s+", " ", s).strip()
        # Avoid breaking KV parsing.
        s = s.replace("；", ",").replace(";", ",")
        s = s.replace("＝", " ").replace("=", " ")
        return s.strip(" ,，")

    def _attach_base_tone(self, instruct: str, max_chars: int) -> str:
        """
        If configured, prefix inferred instruct with a fixed base tone/persona:
        `基调=<...>;情绪=...;语速=...;停顿=...`

        Priority when trimming: keep the inferred instruct fields intact if possible.
        """
        base_raw = self.get_config(ConfigKeys.COMFYUI_AUTO_INSTRUCT_BASE_TONE, "") or ""
        base = self._clean_base_tone(str(base_raw))
        if not base:
            return (instruct or "").strip()

        s = (instruct or "").strip()
        fields = self._parse_instruct_fields(s)
        if "基调" in fields:
            return s

        prefix = f"基调={base}"
        if not s:
            return prefix[:max_chars].rstrip() if max_chars else prefix

        combined = f"{prefix};{s}"
        if not max_chars or len(combined) <= max_chars:
            return combined

        # Too long: try trimming base first, keeping inferred instruct intact.
        remain = max_chars - len(s) - len(";") - len("基调=")
        if remain <= 0:
            # Can't fit base at all; keep instruct (already max_chars-limited upstream).
            return s[:max_chars].rstrip()
        base_trim = base[:remain].rstrip(" ,，")
        return f"基调={base_trim};{s}"

    def _parse_instruct_fields(self, instruct: str) -> Dict[str, str]:
        """
        Parse a 1-line instruct like:
        情绪=愤怒;语速=很快;停顿=很少;表现=咬牙切齿

        We only *use* a few keys (情绪/语速/停顿/强度/表现...), but keep it generic.
        """
        s = (instruct or "").strip()
        if not s:
            return {}

        # Normalize separators (full-width punctuation).
        s = s.replace("；", ";").replace("：", ":").replace("＝", "=")

        # Split by semicolon/comma-like separators.
        parts = [p.strip() for p in re.split(r"[;]+", s) if p.strip()]
        out: Dict[str, str] = {}
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k or not v:
                continue
            # Limit key length to avoid garbage.
            if len(k) > 8:
                continue
            out[k] = v
        return out

    def _map_speed_label(self, label: str) -> Optional[float]:
        lab = (label or "").strip()
        m = {
            "很慢": 0.85,
            "稍慢": 0.93,
            "正常": 1.00,
            "稍快": 1.07,
            "很快": 1.15,
        }
        return m.get(lab)

    def _map_pause_label(self, label: str) -> Optional[float]:
        lab = (label or "").strip()
        m = {
            "很少": 0.6,
            "自然": 1.0,
            "稍多": 1.3,
            "很多": 1.7,
        }
        return m.get(lab)

    def _ensure_base_pause_cfg(self, pause_cfg: Dict[str, float]) -> Dict[str, float]:
        # If caller didn't configure pauses (all zeros), apply a conservative base so "停顿" can take effect.
        keys = ["pause_linebreak", "period_pause", "comma_pause", "question_pause", "hyphen_pause"]
        if all(float(pause_cfg.get(k, 0.0) or 0.0) == 0.0 for k in keys):
            return {
                **pause_cfg,
                "pause_linebreak": 0.18,
                "period_pause": 0.22,
                "comma_pause": 0.10,
                "question_pause": 0.20,
                "hyphen_pause": 0.06,
            }
        return pause_cfg

    def _enrich_instruct_for_emotion(self, instruct: str, max_chars: int) -> str:
        """
        Add short performance cues for common emotions, keeping it single-line KV style.
        This helps when the model under-reacts to simple labels like "愤怒".
        """
        s = (instruct or "").strip()
        if not s:
            return ""

        fields = self._parse_instruct_fields(s)
        emo = fields.get("情绪", "")
        if not emo:
            return s

        # Only add if it doesn't already contain a "表现=" field.
        if "表现" in fields:
            return s

        emo_norm = emo
        cues = ""
        if "愤怒" in emo_norm or "生气" in emo_norm:
            cues = "声压高,咬字重,重音强,尾音下压"
        elif "开心" in emo_norm or "高兴" in emo_norm:
            cues = "笑意明显,轻快上扬,尾音明亮"
        elif "悲伤" in emo_norm or "难过" in emo_norm:
            cues = "气声略多,音量偏低,语尾下沉"
        elif "温柔" in emo_norm:
            cues = "音量轻,气声柔,语尾轻收"
        elif "冷淡" in emo_norm or "冷静" in emo_norm:
            cues = "平直克制,少起伏,干净收尾"

        if not cues:
            return s

        extra = f";表现={cues}"
        if max_chars and len(s) + len(extra) > max_chars:
            # Trim cues to fit.
            allow = max_chars - len(s) - len(";表现=")
            if allow <= 0:
                return s[:max_chars].rstrip()
            cues = cues[:allow].rstrip(",， ")
            extra = f";表现={cues}"
        return (s + extra)[:max_chars].rstrip() if max_chars else (s + extra)

    def _apply_instruct_controls(
        self, instruct: str, speed: float, pause_cfg: Dict[str, float], max_chars: int
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        If instruct contains '语速'/'停顿', map them to real synthesis controls.
        This makes auto_instruct meaningfully affect output even if the model is insensitive to labels.
        """
        s = (instruct or "").strip()
        if not s:
            return "", speed, pause_cfg

        fields = self._parse_instruct_fields(s)
        speed_label = fields.get("语速", "")
        pause_label = fields.get("停顿", "")

        out_speed = float(speed)
        mapped_speed = self._map_speed_label(speed_label)
        if mapped_speed is not None:
            out_speed = mapped_speed

        out_pause_cfg = dict(pause_cfg or {})
        mapped_pause = self._map_pause_label(pause_label)
        if mapped_pause is not None:
            out_pause_cfg = self._ensure_base_pause_cfg(out_pause_cfg)
            for k in ["pause_linebreak", "period_pause", "comma_pause", "question_pause", "hyphen_pause"]:
                try:
                    out_pause_cfg[k] = float(out_pause_cfg.get(k, 0.0) or 0.0) * float(mapped_pause)
                except Exception:
                    pass

        # Add short performance cues (kept within max_chars).
        s = self._enrich_instruct_for_emotion(s, max_chars=max_chars)
        return s, out_speed, out_pause_cfg

    async def _infer_instruct(
        self,
        text: str,
        detected_lang: str,
        chat_stream=None,
        chat_id: Optional[str] = None,
        style_name: str = "",
    ) -> str:
        """
        Infer a short CustomVoice `instruct` string from the target text via MaiBot's LLM interface.
        """
        enabled = bool(self.get_config(ConfigKeys.COMFYUI_AUTO_INSTRUCT_ENABLED, False))
        if not enabled:
            return ""

        max_chars = int(self.get_config(ConfigKeys.COMFYUI_AUTO_INSTRUCT_MAX_CHARS, 40) or 40)

        # Default prompt: output ONE short instruct line only.
        default_tpl = (
            "你是配音导演。请根据要朗读的文本生成一行 TTS instruct。\\n"
            "硬性要求：必须同时包含【情绪】【语速】【停顿】三项。可以额外补充 1-2 个表演提示（如 音量/重音/音高/表现）。\\n"
            "只输出一行，不要解释，不要复述原文，不要引号/代码块。\\n"
            "输出格式固定为：情绪=<...>;语速=<...>;停顿=<...>\\n"
            "语速可选：很慢/稍慢/正常/稍快/很快。\\n"
            "停顿可选：很少/自然/稍多/很多。\\n"
            "长度<= {max_chars} 字。\\n"
            "文本语言: {lang}\\n"
            "待朗读文本: {text}\\n"
        )
        prompt_tpl = str(self.get_config(ConfigKeys.COMFYUI_AUTO_INSTRUCT_PROMPT, default_tpl) or "")
        if not prompt_tpl.strip():
            prompt_tpl = default_tpl

        # Cache key should change if prompt/base_tone/max_chars changes.
        base_raw = str(self.get_config(ConfigKeys.COMFYUI_AUTO_INSTRUCT_BASE_TONE, "") or "")
        cfg_sig_src = f"{max_chars}\\n{prompt_tpl}\\n{base_raw}"
        cfg_sig = hashlib.sha256(cfg_sig_src.encode("utf-8")).hexdigest()[:12]
        text_sig = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        cache_key = f"{cfg_sig}:{detected_lang}:{text_sig}"
        cached = self._instruct_cache.get(cache_key)
        if cached:
            return cached

        lang = detected_lang or "auto"
        prompt = prompt_tpl.format(text=text.strip(), lang=lang, max_chars=max_chars)

        try:
            resp = await generator_api.generate_tts_instruct(
                prompt=prompt,
                request_type="tts_instruct",
            )
            instruct = self._clean_instruct(resp or "", max_chars=max_chars)
            instruct = self._attach_base_tone(instruct, max_chars=max_chars)
            if instruct:
                self._instruct_cache[cache_key] = instruct
            return instruct
        except Exception as e:
            logger.warning(f"{self.log_prefix} auto_instruct 失败(style={style_name}): {e}")
            return ""

    def validate_config(self) -> Tuple[bool, str]:
        server = self.get_config(ConfigKeys.COMFYUI_SERVER, "http://127.0.0.1:8188")
        if not server:
            return False, "ComfyUI 未配置 server"

        input_dir = self.get_config(
            ConfigKeys.COMFYUI_INPUT_DIR,
            "/Users/xenon/Downloads/seiun_tts/ComfyUI/ComfyUI/input",
        )
        if not input_dir:
            return False, "ComfyUI 未配置 input_dir"

        styles_raw = self.get_config(ConfigKeys.COMFYUI_STYLES, {})
        styles = self._normalize_styles_config(styles_raw)
        if not styles:
            return False, "ComfyUI 后端未配置任何风格（至少需要配置 1 个 style）"

        default_name = self.get_default_voice() or "default"
        if default_name not in styles:
            # Fallback to "default" if present.
            if "default" in styles:
                default_name = "default"
            else:
                return False, f"ComfyUI default_style='{default_name}' 不存在"

        st = styles.get(default_name, {})
        mode = (st.get("mode") or "voice_clone").strip()
        if mode == "voice_clone":
            if not st.get("refer_wav") or not st.get("prompt_text"):
                return False, f"ComfyUI 风格 '{default_name}' 配置不完整（voice_clone 需要 refer_wav 和 prompt_text）"
        elif mode == "custom_voice":
            if not st.get("model_path") or not st.get("speaker"):
                return False, f"ComfyUI 风格 '{default_name}' 配置不完整（custom_voice 需要 model_path 和 speaker）"
        else:
            return False, f"ComfyUI 风格 '{default_name}' mode 无效: {mode}"

        return True, ""

    def _ensure_ref_in_input(self, input_dir: str, refer_wav: str) -> str:
        refer_wav = TTSFileManager.resolve_path(refer_wav)
        if not os.path.exists(refer_wav):
            raise FileNotFoundError(f"参考音频不存在: {refer_wav}")

        st = os.stat(refer_wav)
        cache_key = f"{os.path.abspath(refer_wav)}:{st.st_mtime_ns}:{st.st_size}"
        if cache_key in self._ref_cache:
            name = self._ref_cache[cache_key]
            if os.path.exists(os.path.join(input_dir, name)):
                return name

        ext = os.path.splitext(refer_wav)[1] or ".wav"
        h = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
        name = f"maibot_ref_{h}{ext}"
        dst = os.path.join(input_dir, name)

        os.makedirs(input_dir, exist_ok=True)
        if not os.path.exists(dst):
            # Keep it simple: copy file bytes. LoadAudio can decode common formats (wav/mp3).
            import shutil

            shutil.copyfile(refer_wav, dst)

        self._ref_cache[cache_key] = name
        return name

    def _build_prompt_voice_clone(
        self,
        ref_filename: str,
        ref_text: str,
        target_text: str,
        language: str,
        model_choice: str,
        precision: str,
        seed: int,
        max_new_tokens: int,
        top_p: float,
        top_k: int,
        temperature: float,
        repetition_penalty: float,
        audio_quality: str,
        mlx_python: str,
        mlx_cli: str,
        pause_cfg: Dict[str, float],
    ) -> Dict[str, Any]:
        # Node IDs are arbitrary but stable in this prompt template.
        # 1: LoadAudio -> outputs AUDIO
        # 2: Pause config (FB_Qwen3TTSConfig) -> outputs TTS_CONFIG
        # 3: MLX VoiceClone -> outputs AUDIO
        # 4: SaveAudioMP3 -> outputs UI audio file info
        filename_prefix = f"audio/maibot_comfyui_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        prompt: Dict[str, Any] = {
            "1": {
                "class_type": "LoadAudio",
                "inputs": {
                    "audio": ref_filename,
                },
            },
            "2": {
                "class_type": "FB_Qwen3TTSConfig",
                "inputs": {
                    "pause_linebreak": float(pause_cfg.get("pause_linebreak", 0.0)),
                    "period_pause": float(pause_cfg.get("period_pause", 0.0)),
                    "comma_pause": float(pause_cfg.get("comma_pause", 0.0)),
                    "question_pause": float(pause_cfg.get("question_pause", 0.0)),
                    "hyphen_pause": float(pause_cfg.get("hyphen_pause", 0.0)),
                },
            },
            "3": {
                "class_type": "MLX_Qwen3TTSVoiceClone",
                "inputs": {
                    "target_text": target_text,
                    "model_choice": model_choice,
                    "device": "auto",
                    "precision": precision,
                    "language": language,
                    "ref_audio": ["1", 0],
                    "ref_text": ref_text,
                    "seed": int(seed),
                    "max_new_tokens": int(max_new_tokens),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "temperature": float(temperature),
                    "repetition_penalty": float(repetition_penalty),
                    "attention": "auto",
                    "unload_model_after_generate": False,
                    "config": ["2", 0],
                    "mlx_python": mlx_python,
                    "mlx_cli": mlx_cli,
                },
            },
            "4": {
                "class_type": "SaveAudioMP3",
                "inputs": {
                    "audio": ["3", 0],
                    "filename_prefix": filename_prefix,
                    "quality": audio_quality,
                },
            },
        }
        return prompt

    def _build_prompt_custom_voice(
        self,
        target_text: str,
        speaker: str,
        model_path: str,
        instruct: str,
        speed: float,
        language: str,
        seed: int,
        max_new_tokens: int,
        top_p: float,
        top_k: int,
        temperature: float,
        repetition_penalty: float,
        audio_quality: str,
        mlx_python: str,
        mlx_cli: str,
        pause_cfg: Dict[str, float],
    ) -> Dict[str, Any]:
        # 2: Pause config (FB_Qwen3TTSConfig) -> outputs TTS_CONFIG
        # 3: MLX CustomVoice -> outputs AUDIO
        # 4: SaveAudioMP3 -> outputs UI audio file info
        filename_prefix = f"audio/maibot_comfyui_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        prompt: Dict[str, Any] = {
            "2": {
                "class_type": "FB_Qwen3TTSConfig",
                "inputs": {
                    "pause_linebreak": float(pause_cfg.get("pause_linebreak", 0.0)),
                    "period_pause": float(pause_cfg.get("period_pause", 0.0)),
                    "comma_pause": float(pause_cfg.get("comma_pause", 0.0)),
                    "question_pause": float(pause_cfg.get("question_pause", 0.0)),
                    "hyphen_pause": float(pause_cfg.get("hyphen_pause", 0.0)),
                },
            },
            "3": {
                "class_type": "MLX_Qwen3TTSCustomVoice",
                "inputs": {
                    "text": target_text,
                    "speaker": speaker,
                    "model_path": model_path,
                    "instruct": instruct or "",
                    "speed": float(speed),
                    "language": language,
                    "seed": int(seed),
                    "max_new_tokens": int(max_new_tokens),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "temperature": float(temperature),
                    "repetition_penalty": float(repetition_penalty),
                    "config": ["2", 0],
                    "mlx_python": mlx_python,
                    "mlx_cli": mlx_cli,
                },
            },
            "4": {
                "class_type": "SaveAudioMP3",
                "inputs": {
                    "audio": ["3", 0],
                    "filename_prefix": filename_prefix,
                    "quality": audio_quality,
                },
            },
        }
        return prompt

    async def _queue_and_wait(
        self, server: str, prompt: Dict[str, Any], timeout: int
    ) -> Dict[str, Any]:
        session_manager = await TTSSessionManager.get_instance()
        prompt_id = str(uuid.uuid4())

        post_url = f"{server.rstrip('/')}/prompt"
        payload = {
            "prompt": prompt,
            "client_id": "maibot-tts-voice-plugin",
            "prompt_id": prompt_id,
        }

        async with session_manager.post(
            post_url, json=payload, backend_name=self.backend_name, timeout=timeout
        ) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200:
                raise RuntimeError(f"ComfyUI /prompt 失败: {resp.status} {str(data)[:200]}")
            if "error" in data:
                raise RuntimeError(f"ComfyUI /prompt 返回错误: {data['error']}")

        # Poll history until prompt_id appears
        hist_url = f"{server.rstrip('/')}/history/{prompt_id}"
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            async with session_manager.get(
                hist_url, backend_name=self.backend_name, timeout=timeout
            ) as resp:
                history = await resp.json(content_type=None)
            if prompt_id in history:
                return history[prompt_id]
            await asyncio.sleep(0.35)

        raise TimeoutError("等待 ComfyUI 生成超时")

    async def _download_output_audio(self, server: str, history_item: Dict[str, Any], timeout: int) -> bytes:
        outputs = history_item.get("outputs") or {}
        node_out = outputs.get("4") or {}
        audios = node_out.get("audio") or []
        if not audios:
            # Some failures show up only in status/messages.
            status = history_item.get("status") or {}
            raise RuntimeError(f"ComfyUI 未产出音频. status={status}")

        a0 = audios[0]
        filename = a0.get("filename")
        subfolder = a0.get("subfolder", "")
        folder_type = a0.get("type", "output")
        if not filename:
            raise RuntimeError(f"ComfyUI 音频输出结构异常: {a0}")

        q = urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        url = f"{server.rstrip('/')}/view?{q}"

        session_manager = await TTSSessionManager.get_instance()
        async with session_manager.get(url, backend_name=self.backend_name, timeout=timeout) as resp:
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"ComfyUI /view 失败: {resp.status} {txt[:200]}")
            return await resp.read()

    async def execute(self, text: str, voice: Optional[str] = None, **kwargs) -> TTSResult:
        is_valid, err = self.validate_config()
        if not is_valid:
            return TTSResult(False, err, backend_name=self.backend_name)

        if not text or not text.strip():
            return TTSResult(False, "待合成的文本为空", backend_name=self.backend_name)

        server = self.get_config(ConfigKeys.COMFYUI_SERVER, "http://127.0.0.1:8188")
        input_dir = self.get_config(
            ConfigKeys.COMFYUI_INPUT_DIR,
            "/Users/xenon/Downloads/seiun_tts/ComfyUI/ComfyUI/input",
        )
        timeout = int(self.get_config(ConfigKeys.COMFYUI_TIMEOUT, self.get_config(ConfigKeys.GENERAL_TIMEOUT, 60)))

        audio_quality = self.get_config(ConfigKeys.COMFYUI_AUDIO_QUALITY, "128k")
        mlx_python = self.get_config(
            ConfigKeys.COMFYUI_MLX_PYTHON,
            "/Users/xenon/Downloads/seiun_tts/qwen3-tts-apple-silicon/.venv/bin/python",
        )
        mlx_cli = self.get_config(
            ConfigKeys.COMFYUI_MLX_CLI,
            "/Users/xenon/Downloads/seiun_tts/qwen3-tts-apple-silicon/mlx_voice_clone_cli.py",
        )

        styles_raw = self.get_config(ConfigKeys.COMFYUI_STYLES, {})
        styles = self._filter_styles_by_mode(self._normalize_styles_config(styles_raw))

        style_name = (voice or self.get_default_voice() or "").strip() or "default"
        if style_name not in styles:
            # For split backends (voiceclone/customvoice), make "wrong style" errors explicit.
            if (voice or "").strip() and self.allowed_modes:
                return TTSResult(
                    False,
                    f"ComfyUI风格 '{style_name}' 不存在或不属于当前后端({self.backend_name})",
                    backend_name=self.backend_name,
                )
            # Fallback order: "default" -> first available style.
            if "default" in styles:
                style_name = "default"
            elif styles:
                style_name = sorted(styles.keys())[0]
            else:
                return TTSResult(
                    False,
                    f"ComfyUI 未配置任何风格（{self.backend_name}）",
                    backend_name=self.backend_name,
                )
        style = styles.get(style_name, {})

        mode = (style.get("mode") or "voice_clone").strip()
        if mode == "voice_clone":
            refer_wav = style.get("refer_wav", "")
            prompt_text = style.get("prompt_text", "")
            if not refer_wav or not prompt_text:
                return TTSResult(False, f"ComfyUI风格 '{style_name}' 配置不完整（voice_clone）", backend_name=self.backend_name)
        elif mode == "custom_voice":
            model_path = style.get("model_path", "")
            speaker = style.get("speaker", "")
            if not model_path or not speaker:
                return TTSResult(False, f"ComfyUI风格 '{style_name}' 配置不完整（custom_voice）", backend_name=self.backend_name)
        else:
            return TTSResult(False, f"ComfyUI风格 '{style_name}' mode 无效: {mode}", backend_name=self.backend_name)

        # Map language to the MLX node's language combo. Default to Auto.
        detected = TTSTextUtils.detect_language(text)
        language = style.get("language") or LANG_TO_DEMO.get(detected, "Auto")

        # Sampling defaults match the MLX node defaults we exposed.
        seed = int(style.get("seed", 0) or 0)
        model_choice = str(style.get("model_choice", "1.7B") or "1.7B")
        precision = str(style.get("precision", "bf16") or "bf16")
        max_new_tokens = int(style.get("max_new_tokens", 2048) or 2048)
        top_p = float(style.get("top_p", 0.8) or 0.8)
        top_k = int(style.get("top_k", 20) or 20)
        temperature = float(style.get("temperature", 1.0) or 1.0)
        repetition_penalty = float(style.get("repetition_penalty", 1.05) or 1.05)

        pause_cfg = {
            "pause_linebreak": float(self.get_config(ConfigKeys.COMFYUI_PAUSE_LINEBREAK, 0.0)),
            "period_pause": float(self.get_config(ConfigKeys.COMFYUI_PERIOD_PAUSE, 0.0)),
            "comma_pause": float(self.get_config(ConfigKeys.COMFYUI_COMMA_PAUSE, 0.0)),
            "question_pause": float(self.get_config(ConfigKeys.COMFYUI_QUESTION_PAUSE, 0.0)),
            "hyphen_pause": float(self.get_config(ConfigKeys.COMFYUI_HYPHEN_PAUSE, 0.0)),
        }
        # Allow per-style override.
        if isinstance(style.get("pause_cfg"), dict):
            for k in pause_cfg.keys():
                if k in style["pause_cfg"]:
                    try:
                        pause_cfg[k] = float(style["pause_cfg"][k])
                    except Exception:
                        pass

        try:
            if mode == "voice_clone":
                ref_filename = self._ensure_ref_in_input(input_dir, style.get("refer_wav", ""))
                prompt = self._build_prompt_voice_clone(
                    ref_filename=ref_filename,
                    ref_text=style.get("prompt_text", ""),
                    target_text=text,
                    language=language,
                    model_choice=model_choice,
                    precision=precision,
                    seed=seed,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    audio_quality=audio_quality,
                    mlx_python=mlx_python,
                    mlx_cli=mlx_cli,
                    pause_cfg=pause_cfg,
                )
            else:
                # Allow per-style / automatic instruct inference.
                instruct = str(style.get("instruct", "")).strip()
                auto_style = bool(style.get("auto_instruct", False))
                inferred = ""
                if instruct == "__AUTO__" or (not instruct and auto_style):
                    chat_stream = kwargs.get("chat_stream")
                    chat_id = kwargs.get("chat_id")
                    inferred = await self._infer_instruct(
                        text=text,
                        detected_lang=detected,
                        chat_stream=chat_stream,
                        chat_id=chat_id,
                        style_name=style_name,
                    )
                if inferred:
                    instruct = inferred

                # If the instruct contains usable fields, map them to real controls.
                max_chars = int(self.get_config(ConfigKeys.COMFYUI_AUTO_INSTRUCT_MAX_CHARS, 40) or 40)
                instruct, mapped_speed, mapped_pause_cfg = self._apply_instruct_controls(
                    instruct=instruct,
                    speed=float(style.get("speed", 1.0) or 1.0),
                    pause_cfg=pause_cfg,
                    max_chars=max_chars,
                )

                prompt = self._build_prompt_custom_voice(
                    target_text=text,
                    speaker=str(style.get("speaker", "")).strip(),
                    model_path=str(style.get("model_path", "")).strip(),
                    instruct=instruct,
                    speed=mapped_speed,
                    language=language,
                    seed=seed,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    audio_quality=audio_quality,
                    mlx_python=mlx_python,
                    mlx_cli=mlx_cli,
                    pause_cfg=mapped_pause_cfg,
                )

            logger.info(f"{self.log_prefix} ComfyUI请求: text='{text[:50]}...', style={style_name}")
            history_item = await self._queue_and_wait(server, prompt, timeout=timeout)
            audio_bytes = await self._download_output_audio(server, history_item, timeout=timeout)

            ok, msg = TTSFileManager.validate_audio_data(audio_bytes)
            if not ok:
                return TTSResult(False, f"ComfyUI 返回音频无效: {msg}", backend_name=self.backend_name)

            return await self.send_audio(
                audio_data=audio_bytes,
                audio_format="mp3",
                prefix="tts_comfyui",
                voice_info=f"style: {style_name}",
            )
        except Exception as e:
            return TTSResult(False, f"ComfyUI后端错误: {e}", backend_name=self.backend_name)


class ComfyUIVoiceCloneBackend(ComfyUIBackend):
    backend_name = "comfyui_voiceclone"
    backend_description = "ComfyUI 工作流 API（MLX Qwen3-TTS VoiceClone 专用）"
    allowed_modes = {"voice_clone"}

    def get_default_voice(self) -> str:
        v = self.get_config(ConfigKeys.COMFYUI_VOICECLONE_DEFAULT_STYLE, "") or ""
        v = v.strip()
        return v or super().get_default_voice()


class ComfyUICustomVoiceBackend(ComfyUIBackend):
    backend_name = "comfyui_customvoice"
    backend_description = "ComfyUI 工作流 API（MLX Qwen3-TTS CustomVoice 专用）"
    allowed_modes = {"custom_voice"}

    def get_default_voice(self) -> str:
        v = self.get_config(ConfigKeys.COMFYUI_CUSTOMVOICE_DEFAULT_STYLE, "") or ""
        v = v.strip()
        return v or super().get_default_voice()
