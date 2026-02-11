from src.common.server import get_global_server
import os
import importlib.metadata
from maim_message import MessageServer
from src.common.logger import get_logger
from src.config.config import global_config

global_api = None


def get_global_api() -> MessageServer:  # sourcery skip: extract-method
    """获取全局MessageServer实例"""
    global global_api
    if global_api is None:
        # 检查maim_message版本
        try:
            maim_message_version = importlib.metadata.version("maim_message")
            version_int = [int(x) for x in maim_message_version.split(".")]
            version_compatible = version_int >= [0, 3, 3]
            # Check for API Server feature (>= 0.6.0)
            has_api_server_feature = version_int >= [0, 6, 0]
        except (importlib.metadata.PackageNotFoundError, ValueError):
            version_compatible = False
            has_api_server_feature = False

        # 读取配置项
        maim_message_config = global_config.maim_message

        # 设置基本参数 (Legacy Server Mode)
        kwargs = {
            "host": os.environ["HOST"],
            "port": int(os.environ["PORT"]),
            "app": get_global_server().get_app(),
        }

        # 只有在版本 >= 0.3.0 时才使用高级特性
        if version_compatible:
            # 添加自定义logger
            maim_message_logger = get_logger("maim_message")
            kwargs["custom_logger"] = maim_message_logger

            # 添加token认证
            if maim_message_config.auth_token and len(maim_message_config.auth_token) > 0:
                kwargs["enable_token"] = True

            # Removed legacy custom config block (use_custom) as requested.
            kwargs["enable_custom_uvicorn_logger"] = False

        global_api = MessageServer(**kwargs)
        if version_compatible and maim_message_config.auth_token:
            for token in maim_message_config.auth_token:
                global_api.add_valid_token(token)

        # ---------------------------------------------------------------------
        # Additional API Server Configuration (maim_message >= 6.0)
        # ---------------------------------------------------------------------
        enable_api_server = maim_message_config.enable_api_server
        
        # 如果版本支持且启用了API Server，则初始化额外服务器
        if has_api_server_feature and enable_api_server:
            try:
                from maim_message.server import WebSocketServer, ServerConfig
                from maim_message.message import APIMessageBase

                api_logger = get_logger("maim_message_api_server")

                # 1. Prepare Config
                api_server_host = maim_message_config.api_server_host
                api_server_port = maim_message_config.api_server_port
                use_wss = maim_message_config.api_server_use_wss

                server_config = ServerConfig(
                    host=api_server_host,
                    port=api_server_port,
                    ssl_enabled=use_wss,
                    ssl_certfile=maim_message_config.api_server_cert_file if use_wss else None,
                    ssl_keyfile=maim_message_config.api_server_key_file if use_wss else None,
                )

                # 2. Setup Auth Handler
                async def auth_handler(metadata: dict) -> bool:
                    allowed_keys = maim_message_config.api_server_allowed_api_keys
                    # If list is empty/None, allow all (default behavior of returning True)
                    if not allowed_keys:
                        return True
                    
                    api_key = metadata.get("api_key")
                    if api_key in allowed_keys:
                        return True
                    
                    api_logger.warning(f"Rejected connection with invalid API Key: {api_key}")
                    return False

                server_config.on_auth = auth_handler

                # 3. Setup Message Bridge
                # Initialize refined route map if not exists
                if not hasattr(global_api, "platform_map"):
                    global_api.platform_map = {}

                async def bridge_message_handler(message: APIMessageBase, metadata: dict):
                    # Bridge message to the main bot logic
                    # We convert APIMessageBase to dict to be compatible with legacy handlers
                    # that MainBot (ChatManager) expects.
                    msg_dict = message.to_dict()
                    
                    # Compatibility Layer: Flatten sender_info to top-level user_info/group_info
                    # Legacy MessageBase expects message_info to have user_info and group_info directly.
                    if "message_info" in msg_dict:
                        msg_info = msg_dict["message_info"]
                        sender_info = msg_info.get("sender_info")
                        if sender_info:
                            # If direct user_info/group_info are missing, populate them from sender_info
                            if "user_info" not in msg_info and (ui := sender_info.get("user_info")):
                                msg_info["user_info"] = ui
                            
                            if "group_info" not in msg_info and (gi := sender_info.get("group_info")):
                                msg_info["group_info"] = gi
                                
                        # Route Caching Logic: Simply map platform to API Key
                        # This allows us to send messages back to the correct API client for this platform
                        try:
                            api_key = metadata.get("api_key")
                            if api_key:
                                platform = msg_info.get("platform")
                                if platform:
                                    global_api.platform_map[platform] = api_key
                        except Exception as e:
                            api_logger.warning(f"Failed to update platform map: {e}")

                    # Compatibility Layer: Ensure raw_message exists (even if None) as it's part of MessageBase
                    if "raw_message" not in msg_dict:
                        msg_dict["raw_message"] = None
                    
                    await global_api.process_message(msg_dict)

                server_config.on_message = bridge_message_handler

                # 4. Initialize Server
                extra_server = WebSocketServer(config=server_config)
                
                # 5. Patch global_api lifecycle methods to manage both servers
                original_run = global_api.run
                original_stop = global_api.stop

                async def patched_run():
                    api_logger.info(f"Starting Additional API Server on {api_server_host}:{api_server_port} (WSS: {use_wss})")
                    # Start the extra server (non-blocking start)
                    await extra_server.start()
                    # Run the original legacy server (this usually keeps running)
                    await original_run()

                async def patched_stop():
                    api_logger.info("Stopping Additional API Server...")
                    await extra_server.stop()
                    await original_stop()

                global_api.run = patched_run
                global_api.stop = patched_stop
                
                # Attach for reference
                global_api.extra_server = extra_server

            except ImportError:
                get_logger("maim_message").error("Cannot import maim_message.server components. Is maim_message >= 0.6.0 installed?")
            except Exception as e:
                get_logger("maim_message").error(f"Failed to initialize Additional API Server: {e}")
                import traceback
                get_logger("maim_message").debug(traceback.format_exc())

    return global_api
