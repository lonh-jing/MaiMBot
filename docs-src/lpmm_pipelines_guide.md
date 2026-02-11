## LPMM 知识库流水线使用指南（命令行版）

本文档介绍如何使用 `scripts/lpmm_manager.py` 及相关子脚本，完成 **导入 / 删除 / 自检 / 刷新 / 回归测试** 等常见流水线操作，并说明各参数在交互式与非交互（脚本化）场景下的用法。

所有命令均假设在项目根目录 `MaiBot/` 下执行：

```bash
cd MaiBot
```

---

## 1. 管理脚本总览：`scripts/lpmm_manager.py`

### 1.1 基本用法

```bash
python scripts/lpmm_manager.py [--interactive] [-a ACTION] [--non-interactive] [-- ...子脚本参数...]
```

- `--interactive` / `-i`：进入交互式菜单模式（推荐人工运维时使用）。
- `--action` / `-a`：直接执行指定操作（非交互入口），可选值：
  - `prepare_raw`：预处理 `data/lpmm_raw_data/*.txt`。
  - `info_extract`：信息抽取，生成 OpenIE JSON 批次。
  - `import_openie`：导入 OpenIE 批次到向量库与知识图。
  - `delete`：删除/回滚知识（封装 `delete_lpmm_items.py`）。
  - `batch_inspect`：检查指定 OpenIE 批次的存在情况。
  - `global_inspect`：全库状态统计。
  - `refresh`：刷新 LPMM 磁盘数据到内存。
  - `test`：检索效果回归测试。
  - `full_import`：一键执行「预处理原始语料 → 信息抽取 → 导入 → 刷新」。
- `--non-interactive`：
  - 启用 **非交互模式**：`lpmm_manager` 自身不会再调用 `input()` 询问确认；
  - 同时自动向子脚本透传 `--non-interactive`（若子脚本支持），用于在 CI / 定时任务中实现无人值守。
- `--` 之后的内容会原样传递给对应子脚本的 `main()`，用于设置更细粒度参数。

> 注意：`--interactive` 与 `--non-interactive` 互斥，不能同时使用。

---

## 2. 典型流水线一：全量导入（从原始 txt 到可用 LPMM）

### 2.1 前置条件

- 将待导入的原始文本放入：

```text
data/lpmm_raw_data/*.txt
```

- 文本按「空行分段」，每个段落为一条候选知识。

### 2.2 一键全流程（交互式）

```bash
python scripts/lpmm_manager.py --interactive
```

菜单中依次：

1. 选择 `9. full_import`（预处理 → 信息抽取 → 导入 → 刷新）。
2. 按提示确认可能的费用与时间消耗。
3. 等待脚本执行完成。

### 2.3 一键全流程（非交互 / CI 友好）

```bash
python scripts/lpmm_manager.py -a full_import --non-interactive
```

执行顺序：

1. `prepare_raw`：调用 `raw_data_preprocessor.load_raw_data()`，统计段落与去重哈希数。
2. `info_extract`：调用 `info_extraction.main(--non-interactive)`，从 `data/lpmm_raw_data` 读取段落，生成 OpenIE JSON 并写入 `data/openie/`。
3. `import_openie`：调用 `import_openie.main(--non-interactive)`，导入 OpenIE 批次到嵌入库与 KG。
4. `refresh`：调用 `refresh_lpmm_knowledge.main()`，刷新 LPMM 知识库到内存。

在 `--non-interactive` 模式下：

- 若 `data/lpmm_raw_data` 中没有 `.txt` 文件，或 `data/openie` 中没有 `.json` 文件，将直接报错退出，并在日志中说明缺少的目录/文件。
- 若 OpenIE 批次中存在非法文段，导入脚本会 **直接报错退出**，不会卡在交互确认上。

---

## 3. 典型流水线二：分步导入

若需要逐步调试或只执行部分步骤，可以分开调用：

### 3.1 预处理原始语料：`prepare_raw`

```bash
python scripts/lpmm_manager.py -a prepare_raw
```

行为：
- 使用 `raw_data_preprocessor.load_raw_data()` 读取 `data/lpmm_raw_data/*.txt`；
- 输出段落总数与去重后的哈希数，供人工检查原始数据质量。

### 3.2 信息抽取：`info_extract`

#### 交互式（带费用提示）

```bash
python scripts/lpmm_manager.py -a info_extract
```

脚本会：
- 打印预计费用/时间提示；
- 询问 `确认继续执行？(y/n)`；
- 然后开始从 `data/lpmm_raw_data` 中读取段落，调用 LLM 提取实体与三元组，并生成 OpenIE JSON。

#### 非交互式（无人工确认）

```bash
python scripts/lpmm_manager.py -a info_extract --non-interactive
```

行为差异：
- 跳过`确认继续执行`的交互提示，直接开始抽取；
- 若 `data/lpmm_raw_data` 下没有 `.txt` 文件，会打印告警并以错误方式退出。

### 3.3 导入 OpenIE 批次：`import_openie`

#### 交互式

```bash
python scripts/lpmm_manager.py -a import_openie
```

脚本会：
- 提示导入开销与资源占用情况；
- 询问是否继续；
- 调用 `OpenIE.load()` 加载批次，再将其导入嵌入库与 KG。

#### 非交互式

```bash
python scripts/lpmm_manager.py -a import_openie --non-interactive
```

- 跳过导入开销确认；
- 若数据存在非法文段：
  - 在交互模式下会询问是否删除这些非法文段并继续；
  - 在非交互模式下，会直接 `logger.error` 并 `sys.exit(1)`，防止导入不完整数据。

> 提示：当前 `OpenIE.load()` 仍可能在内部要求你选择具体批次文件，若需完全无交互的导入，可后续扩展为显式指定文件路径。

### 3.4 刷新 LPMM 知识库：`refresh`

```bash
python scripts/lpmm_manager.py -a refresh
# 或
python scripts/lpmm_manager.py -a refresh --non-interactive
```

两者行为相同：
- 调用 `refresh_lpmm_knowledge.main()`，内部执行 `lpmm_start_up()`；
- 日志中输出当前向量与 KG 规模，验证导入是否成功。

---

## 4. 典型流水线三：删除 / 回滚

删除操作通过 `lpmm_manager.py -a delete` 封装 `scripts/delete_lpmm_items.py`。

### 4.1 交互式删除（推荐人工操作）

```bash
python scripts/lpmm_manager.py --interactive
```

菜单中选择：

1. `4. delete - 删除/回滚知识`
2. 再选择删除方式：
   - 按哈希文件（`--hash-file`）
   - 按 OpenIE 批次（`--openie-file`）
   - 按原始语料 + 段落索引（`--raw-file + --raw-index`）
   - 按关键字搜索现有段落（`--search-text`）
3. 管理脚本会根据你的选择自动拼好常用参数（是否删除实体/关系、是否删除孤立实体、是否 dry-run、是否自动确认等），最后调用 `delete_lpmm_items.py` 执行。

### 4.2 非交互删除（CI / 脚本场景）

#### 示例：按哈希文件删除（带完整保护参数）

```bash
python scripts/lpmm_manager.py -a delete --non-interactive -- \
  --hash-file data/lpmm_delete_hashes.txt \
  --delete-entities \
  --delete-relations \
  --remove-orphan-entities \
  --max-delete-nodes 2000 \
  --yes
```

- `--non-interactive`（manager）：禁止任何 `input()` 询问；
- 子脚本 `delete_lpmm_items.py` 中：
  - `--hash-file`：指定待删段落哈希列表；
  - `--delete-entities` / `--delete-relations` / `--remove-orphan-entities`：同步清理实体与关系；
  - `--max-delete-nodes`：单次删除节点数上限，避免误删过大规模；
  - `--yes`：跳过终极确认，适合已验证的自动流水线。

#### 按 OpenIE 批次删除（常用于批次回滚）

```bash
python scripts/lpmm_manager.py -a delete --non-interactive -- \
  --openie-file data/openie/2025-01-01-12-00-openie.json \
  --delete-entities \
  --delete-relations \
  --remove-orphan-entities \
  --yes
```

### 4.3 非交互模式下的安全限制

在 `delete_lpmm_items.py` 中：

- 若使用 `--search-text`，需要用户通过输入序号选择要删条目；
  - 在 `--non-interactive` 模式下，这一步会直接报错退出，提示改用 `--hash-file / --openie-file / --raw-file` 等纯参数方式。
- 若未指定 `--yes`：
  - 非交互模式下会报错退出，提示「非交互模式且未指定 --yes，出于安全考虑删除操作已被拒绝」。

---

## 5. 典型流水线四：自检与状态检查

### 5.1 检查指定 OpenIE 批次状态：`batch_inspect`

```bash
python scripts/lpmm_manager.py -a batch_inspect -- --openie-file data/openie/xx.json
```

输出该批次在当前库中的：
- 段落向量数量 / KG 段落节点数量；
- 实体向量数量 / KG 实体节点数量；
- 关系向量数量；
- 少量仍存在的样例内容。

常用于：
- 导入后确认是否完全成功；
- 删除后确认是否完全回滚。

### 5.2 查看整库状态：`global_inspect`

```bash
python scripts/lpmm_manager.py -a global_inspect
```

输出：
- 段落 / 实体 / 关系向量条数；
- KG 节点/边总数，段落节点数、实体节点数；
- 实体计数表 `ent_appear_cnt` 的条目数；
- 少量剩余段落/实体样例，便于快速 sanity check。

---

## 6. 典型流水线五：检索效果回归测试

### 6.1 使用默认测试用例

```bash
python scripts/lpmm_manager.py -a test
```

- 调用 `test_lpmm_retrieval.py` 内置的 `DEFAULT_TEST_CASES`；
- 对每条用例输出：
  - 原始结果；
  - 状态（`PASS` / `WARN` / `NO_HIT` / `ERROR`）；
  - 期望关键字与命中关键字列表。

### 6.2 自定义测试问题与期望关键字

```bash
python scripts/lpmm_manager.py -a test -- --query "LPMM 是什么？" \
  --expect-keyword 哈希列表 \
  --expect-keyword 删除脚本
```

也可以直接调用子脚本：

```bash
python scripts/test_lpmm_retrieval.py \
  --query "LPMM 是什么？" \
  --expect-keyword 哈希列表 \
  --expect-keyword 删除脚本
```

---

## 7. 推荐组合示例

### 7.1 导入 + 刷新 + 简单回归

```bash
# 1. 执行全量导入（支持非交互）
python scripts/lpmm_manager.py -a full_import --non-interactive

# 2. 使用内置用例做一次检索回归
python scripts/lpmm_manager.py -a test
```

### 7.2 批次回滚 + 自检

```bash
TARGET_BATCH=data/openie/2025-01-01-12-00-openie.json

# 1. 按批次删除（非交互）
python scripts/lpmm_manager.py -a delete --non-interactive -- \
  --openie-file "$TARGET_BATCH" \
  --delete-entities \
  --delete-relations \
  --remove-orphan-entities \
  --yes

# 2. 检查该批次是否彻底删除
python scripts/lpmm_manager.py -a batch_inspect -- --openie-file "$TARGET_BATCH"

# 3. 查看全库状态
python scripts/lpmm_manager.py -a global_inspect
```

---

如需扩展更多流水线（例如「导入特定批次后自动跑自定义测试用例」），可以在 `scripts/lpmm_manager.py` 中新增对应的 `ACTION_INFO` 条目和 `run_action` 分支，或直接在 CI / shell 脚本中串联上述命令。该管理脚本已支持参数化与非交互调用，适合作为二次封装的基础入口。 


