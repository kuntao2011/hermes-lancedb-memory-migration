---
name: lancedb-memory-migration
description: "Migrate a Hermes profile's session history from FTS5 (built-in session search) to LanceDB vector memory. Use when switching a profile (especially sub-agent profiles) to the LanceDB-backed memory system. Handles state.db → LanceDB migration, config changes, and verification. Requires: Ollama with bge-m3:567m, lancedb pip package."
version: 1.0.0
author: Hermes Agent + Kuntao
license: MIT
metadata:
  hermes:
    tags: [memory, lancedb, ollama, migration, sub-agent, vector, fts5]
---

# LanceDB Memory Migration

将 Hermes profile（通常是子 agent profile）的记忆系统从 FTS5（内置会话搜索）迁移到 LanceDB 向量搜索（LanceDB + Ollama 向量）。

## When to Use

- 已有 profile 的历史 session（存在 `state.db` 中）想从 FTS5 会话搜索切换到 LanceDB 向量搜索
- 新建子 agent profile 时，直接配置使用 LanceDB 向量记忆系统（无需迁移）
- 子 agent profile 需要和 default agent 共用同一套记忆架构

## Architecture Overview

```
Session History (state.db)
        ↓  [迁移脚本]
LanceDB Vector DB  ←  Ollama bge-m3:567m
        ↓  [lancedb-embed plugin]
Hermes Agent (vec_memory_* tools)
```

**三层存储：**

| 层 | 组件 | 作用 |
|----|------|------|
| 原始数据 | `state.db` (SQLite) | 原始 session 消息 |
| 向量嵌入 | Ollama + bge-m3:567m | 文本→1024维向量 |
| 向量数据库 | LanceDB (.lance 文件) | ANN 索引检索 |

**两种 profile 场景：**

| 场景 | HERMES_HOME | LanceDB 路径 |
|------|------------|-------------|
| default agent | `~/.hermes/` | `~/.hermes/lance_memory/` |
| 子 agent profile | `~/.hermes/profiles/<name>/` | `~/.hermes/profiles/<name>/lance_memory/` |

> 子 agent 是进程内线程，与父 agent 共享 HERMES_HOME（各自配置文件里的）。因此子 agent 的 LanceDB 路径自动指向各自 profile 下的 `lance_memory/` 目录。

## Prerequisites

### 1. Ollama 服务 + bge-m3:567m 模型

```bash
# 检查 Ollama 是否运行
bash scripts/check_ollama.sh

# 如果没有，拉取模型
ollama pull bge-m3:567m
```

### 2. lancedb Python 包

```bash
uv pip install --python ~/.hermes/venv/bin/python lancedb
```

## Step 1: 创建迁移脚本

在 `~/.hermes/scripts/` 下创建迁移脚本，文件名格式：`migrate_<profile>_sessions_to_lancedb.py`

**需要修改的配置项（在脚本顶部）：**

```python
PROFILE = "<你的profile名>"           # 例如: "chip_expert", "financial_expert"
OLLAMA_HOST = "http://localhost:11434"  # 根据你的环境调整
OLLAMA_MODEL = "bge-m3:567m"            # Ollama 中的模型名，需与实际一致
VECTOR_DIM = 1024                        # bge-m3:567m 输出维度，勿改动
MIN_ASST_CHARS = 200                     # assistant 内容少于这个字符的 session 跳过
```

**完整脚本模板：**

```python
#!/usr/bin/env python3
"""
Migrate session history from <PROFILE>'s state.db to LanceDB vector memory.

Usage:
    python migrate_<profile>_sessions_to_lancedb.py [--dry-run]
"""

import argparse
import sqlite3
import time
from pathlib import Path
import numpy as np
import requests
import json
import uuid

# ─── Config ───────────────────────────────────────────────────────────────────
# TODO: 修改以下配置项
PROFILE = "<profile名>"                                    # 例如: "chip_expert"
HERMES_HOME = Path.home() / ".hermes"
PROFILE_HOME = HERMES_HOME / "profiles" / PROFILE
OLLAMA_HOST = "http://localhost:11434"                     # 根据环境调整
OLLAMA_MODEL = "bge-m3:567m"                               # 确保与 Ollama 模型名一致
VECTOR_DIM = 1024                                           # bge-m3:567m 维度，勿改动
MIN_ASST_CHARS = 200                                        # assistant 内容少于此值则跳过

# ─── Ollama embed via HTTP API ────────────────────────────────────────────────
def ollama_embed(texts: list[str]) -> list[np.ndarray]:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": OLLAMA_MODEL, "input": texts},
        timeout=300,  # 大 session 可能需要较长时间
    )
    resp.raise_for_status()
    return [np.array(e, dtype=np.float32) for e in resp.json()["embeddings"]]

def check_ollama() -> bool:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if OLLAMA_MODEL not in models:
            print(f"WARNING: Model '{OLLAMA_MODEL}' not loaded.")
            print(f"  Loaded: {models}")
            return False
        return True
    except Exception as e:
        print(f"ERROR: Cannot reach Ollama: {e}")
        return False

# ─── LanceDB Schema ───────────────────────────────────────────────────────────
def build_schema():
    import pyarrow as pa
    return pa.schema([
        pa.field("id",         pa.string()),
        pa.field("content",    pa.string()),
        pa.field("role",       pa.string()),
        pa.field("session_id", pa.string()),
        pa.field("vector",     pa.list_(pa.float32(), VECTOR_DIM)),
        pa.field("created_at", pa.float64()),
        pa.field("metadata",   pa.string()),
    ])

def get_or_create_lance_db():
    import lancedb
    lance_dir = PROFILE_HOME / "lance_memory"
    lance_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lance_dir))
    table_name = "memories"
    if table_name not in db.table_names():
        db.create_table(table_name, schema=build_schema())
        print(f"  Created: {lance_dir}")
    return db, table_name

def get_existing_session_ids(table) -> set[str]:
    try:
        all_rows = table.search([0.0] * VECTOR_DIM).limit(10000).to_list()
        return {r["session_id"] for r in all_rows if r.get("session_id")}
    except Exception:
        return set()

# ─── Session processor (single-pass) ─────────────────────────────────────────
# TODO: 根据需要调整过滤关键词（应与你的语言环境匹配）
SHORT_PATTERNS = [
    "你好", "您好", "hello",
    "你是什么模型", "你现在用的什么模型", "现在你是什么模型",
    "有什么我可以帮", "有什么能帮",
]

def process_session(raw_rows: list, session_id: str, started_at: float,
                    start_dt: str, msg_count: int) -> dict | None:
    """
    单次遍历：同时判断是否有价值并提取内容。
    返回 None 表示不值得迁移，否则返回迁移数据字典。

    内容格式与原始 Ollama embed 写入格式一致：
    [user]
    xxx
    [assistant]
    yyy

    [user]
    xxx
    [assistant]
    yyy
    ...
    """
    user_msgs, asst_msgs = [], []
    for role, content in raw_rows:
        if role not in ('user', 'assistant'):
            continue
        if not content or not content.strip():
            continue
        if role == 'user' and len(content.strip()) < 5:
            continue
        if role == 'user':
            user_msgs.append(content.strip())
        else:
            asst_msgs.append(content.strip())

    if not user_msgs or not asst_msgs:
        return None
    if all(len(m) < 10 for m in user_msgs):
        return None
    pure_confirmation = all(
        any(p in m for p in SHORT_PATTERNS) and len(m) < 30
        for m in user_msgs
    )
    if pure_confirmation:
        return None
    total_asst = sum(len(m) for m in asst_msgs)
    if total_asst < MIN_ASST_CHARS:
        return None

    # 按 user/assistant 顺序交替拼接
    lines = []
    for user_c, asst_c in zip(user_msgs, asst_msgs):
        lines.append(f"[user]\n{user_c}")
        lines.append(f"[assistant]\n{asst_c}")
    if len(user_msgs) > len(asst_msgs):
        for content in user_msgs[len(asst_msgs):]:
            lines.append(f"[user]\n{content}")
            lines.append("[assistant]\n")

    content = "\n\n".join(lines)
    if len(content) < 50:
        return None

    return {
        "session_id": session_id,
        "started_at": started_at,
        "start_dt": start_dt,
        "msg_count": msg_count,
        "asst_chars": total_asst,
        "content": content,
        "preview": content[:200].replace("\n", " "),
    }

# ─── Migration ─────────────────────────────────────────────────────────────────
def migrate(dry_run: bool = False):
    t0 = time.time()
    state_db = PROFILE_HOME / "state.db"
    if not state_db.exists():
        print(f"ERROR: state.db not found: {state_db}")
        return

    print("=" * 60)
    print("Session History Migration: state.db → LanceDB")
    print(f"Profile: {PROFILE}")
    print("=" * 60)
    print(f"  Source:   {state_db}")
    print(f"  Ollama:   {OLLAMA_HOST}")
    print(f"  Model:    {OLLAMA_MODEL}")
    print(f"  Min asst: {MIN_ASST_CHARS} chars")
    print(f"  Dry run:  {dry_run}\n")

    if not check_ollama():
        print("ERROR: Ollama check failed. Aborting.")
        return

    conn = sqlite3.connect(str(state_db))
    cur = conn.cursor()
    cur.execute("""
        SELECT id, message_count, started_at,
               datetime(started_at, 'unixepoch') as start_dt
        FROM sessions WHERE message_count > 0 ORDER BY started_at
    """)
    sessions = cur.fetchall()
    print(f"Found {len(sessions)} sessions with messages.\n")

    db, table_name = get_or_create_lance_db()
    table = db.open_table(table_name)
    existing_ids = get_existing_session_ids(table)
    print(f"LanceDB already has {len(existing_ids)} sessions.\n")

    sessions_to_migrate, skipped_already, skipped_short = [], 0, 0
    for sid, msg_count, started_at, start_dt in sessions:
        if sid in existing_ids:
            skipped_already += 1
            continue
        cur.execute("""
            SELECT role, content FROM messages
            WHERE session_id = ? ORDER BY timestamp
        """, (sid,))
        raw = cur.fetchall()
        result = process_session(raw, sid, started_at, start_dt, msg_count)
        if result:
            sessions_to_migrate.append(result)
        else:
            skipped_short += 1
    conn.close()

    print(f"=== Summary ===")
    print(f"  Total sessions:    {len(sessions)}")
    print(f"  Already migrated: {skipped_already}")
    print(f"  Skipped (short): {skipped_short}")
    print(f"  To migrate:       {len(sessions_to_migrate)}\n")

    if not sessions_to_migrate:
        print("No new sessions to migrate.")
        return

    print("=== Sessions ===\n")
    for s in sessions_to_migrate:
        print(f"  {s['session_id']}  started:{s['start_dt']}  "
              f"asst:{s['asst_chars']}chars  preview:{s['preview'][:80]}...")
        print()

    if dry_run:
        print(f"[DRY RUN] Would migrate {len(sessions_to_migrate)} sessions.")
        return

    print(f"=== Migrating {len(sessions_to_migrate)} sessions ===\n")
    migrated, errors = 0, 0
    for batch_start in range(0, len(sessions_to_migrate), 4):
        batch = sessions_to_migrate[batch_start:batch_start + 4]
        texts = [s["content"] for s in batch]
        try:
            embeddings = ollama_embed(texts)
        except Exception as e:
            print(f"  ERROR embedding: {e}")
            errors += len(batch)
            continue

        now = time.time()
        rows = []
        for s, emb in zip(batch, embeddings):
            rows.append({
                "id": str(uuid.uuid4()),
                "content": s["content"],
                "role": "session_migrated",
                "session_id": s["session_id"],
                "vector": emb.tolist(),
                "created_at": now,
                "metadata": json.dumps({
                    "started_at": s["start_dt"],
                    "msg_count": s["msg_count"],
                    "asst_chars": s["asst_chars"],
                }),
            })
        try:
            table.add(rows)
            migrated += len(rows)
            print(f"  [OK] Batch {batch_start}-{batch_start+len(batch)}: "
                  f"{len(rows)} rows written")
        except Exception as e:
            print(f"  ERROR writing: {e}")
            errors += len(batch)

    print(f"\n{'='*60}")
    print(f"Migration Complete")
    print(f"  Migrated: {migrated}  Errors: {errors}  Time: {time.time()-t0:.1f}s")
    print(f"  LanceDB total: {table.count_rows()} rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    migrate(dry_run=parser.parse_args().dry_run)
```

## Step 2: 修改 config.yaml

在目标 profile 的 `config.yaml` 中找到或添加 `memory` 区块：

```yaml
# Memory — 改为 lancedb-embed
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
  provider: lancedb-embed
```

**如何定位：**
```bash
grep -n "^memory:" ~/.hermes/profiles/<profile>/config.yaml
```

**如何修改（使用 hermes config 或直接编辑）：**
```bash
hermes config set memory.provider lancedb-embed --profile <profile>
```

> 如果使用 hermes CLI 直接改，不需要重启 gateway。但如果直接编辑 config.yaml，需要重启该 profile 的 gateway。

## Step 3: 配置 lancedb-embed 插件（可选）

`lancedb-embed` 插件会从 `plugins.lancedb-embed` 配置块读取参数。如果不配置，则使用硬编码默认值。

**默认值：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_url` | `http://localhost:11434` | Ollama 服务地址 |
| `embedding_model` | `bge-m3:567m` | 嵌入模型名 |
| `lance_dir` | `$HERMES_HOME/lance_memory` | LanceDB 目录（自动解析） |
| `batch_size` | `32` | 每批嵌入数量 |
| `search_top_k` | `5` | 搜索返回数量 |
| `min_content_len` | `50` | 最小内容长度 |

**如果需要自定义，在 profile 的 config.yaml 中添加：**

```yaml
plugins:
  lancedb-embed:
    base_url: http://localhost:11434      # 根据你的环境调整
    embedding_model: bge-m3:567m           # 确保与 Ollama 中实际模型名一致
    lance_dir: $HERMES_HOME/lance_memory   # 自动解析为 profile 的 HERMES_HOME
    batch_size: 32
    search_top_k: 5
```

## Step 4: 执行迁移

### 4.1 Dry-run 验证

```bash
~/.hermes/venv/bin/python3 ~/.hermes/scripts/migrate_<profile>_sessions_to_lancedb.py --dry-run
```

预期输出：
- 显示找到的 session 数量
- 显示"Already migrated"数量（首次为0）
- 列出将被迁移的 session 及其信息

### 4.2 执行迁移

```bash
~/.hermes/venv/bin/python3 ~/.hermes/scripts/migrate_<profile>_sessions_to_lancedb.py
```

> 大 session（>20000字符）的嵌入可能需要 60-300 秒，受 Ollama 模型处理速度影响。

### 4.3 幂等性

脚本会自动跳过已迁移的 session（通过 session_id 去重）。可以多次运行，不会重复添加。

## Step 5: 全面验证

### 5.1 配置检查

```bash
# 检查 memory.provider
grep -n "provider:" ~/.hermes/profiles/<profile>/config.yaml | grep memory

# 期望输出包含: provider: lancedb-embed
```

### 5.2 LanceDB 数据检查

```bash
# 检查 LanceDB 目录和文件
ls -la ~/.hermes/profiles/<profile>/lance_memory/memories.lance
du -sh ~/.hermes/profiles/<profile>/lance_memory/

# 检查行数
~/.hermes/venv/bin/python3 -c "
import lancedb
db = lancedb.connect('~/.hermes/profiles/<profile>/lance_memory')
table = db.open_table('memories')
print(f'Rows: {table.count_rows()}')
"
```

### 5.3 Ollama 模型检查

```bash
bash scripts/check_ollama.sh
```

### 5.4 向量搜索联动测试

```bash
~/.hermes/venv/bin/python3 -c "
import lancedb, json, requests, numpy as np

OLLAMA_HOST = 'http://localhost:11434'
OLLAMA_MODEL = 'bge-m3:567m'
LANCE_DIR = '~/.hermes/profiles/<profile>/lance_memory'

# 获取查询向量
texts = ['<与profile相关的测试查询词>']
resp = requests.post(f'{OLLAMA_HOST}/api/embed', json={'model': OLLAMA_MODEL, 'input': texts}, timeout=60)
emb = np.array(resp.json()['embeddings'][0], dtype=np.float32)

# 搜索
db = lancedb.connect(LANCE_DIR)
table = db.open_table('memories')
results = table.search(emb).limit(3).to_list()
print(f'Found {len(results)} results:')
for r in results:
    meta = json.loads(r.get('metadata', '{}'))
    print(f'  {r[\"session_id\"]} dist={r.get(\"_distance\",\"N/A\")} asst={meta.get(\"asst_chars\",\"?\")}')
"
```

### 5.5 Gateway 重启（如果需要）

如果 gateway 正在运行且使用了旧的配置，重启使新配置生效：

```bash
# 查找该 profile 的 gateway service 名称
systemctl --user list-units | grep hermes

# 重启（将 <profile> 替换为实际名称）
systemctl --user restart hermes-gateway-<profile>.service
```

## Pitfalls

### 1. Ollama 模型名不匹配

错误：`WARNING: Model 'bge-m3:567m' not loaded`

原因：Ollama 中注册的模型名与你脚本里写的不一致。

解决：
```bash
# 查看 Ollama 中实际注册的模型名
bash scripts/check_ollama.sh | grep -q "OK" && ollama list | grep "bge-m3:567m"
```
将脚本中 `OLLAMA_MODEL` 改为实际模型名。

### 2. 子 agent 默认不使用 memory

子 agent（`delegate_task`）默认 `skip_memory=True`，跳过了 memory provider 加载。如需子 agent 主动使用向量搜索，需要修改 `delegate_tool.py`：

```python
# tools/delegate_tool.py 第 1105 行附近
# 改 skip_memory=True 为 skip_memory=False
# 同时 DEFAULT_TOOLSETS 加上 "memory"
```

> 注意：这会影响所有子 agent。改动前确认你的需求。

### 3. 向量嵌入超时

大 session（>20000字符）嵌入可能超时。脚本已将 timeout 设为 300 秒。如果仍超时，可能是 Ollama 处理速度慢或网络问题。

### 4. content 拼接格式（与原始 Ollama embed 格式保持一致）

脚本中的拼接逻辑：按 user/assistant 顺序交替拼接，每一对是 `[user]\n用户内容\n[assistant]\n助手内容`。这样生成的内容与 Ollama embed 写入的原始格式一致，搜索时语义一致。

### 5. Profile 独立性问题

各 profile 的 LanceDB 完全独立。如果多个 profile 想共用同一份记忆数据，需要手动 symlink 或统一 `lance_dir` 配置。

## File Locations

| 文件 | 路径 |
|------|------|
| 迁移脚本 | `~/.hermes/scripts/migrate_<profile>_sessions_to_lancedb.py` |
| Profile config | `~/.hermes/profiles/<profile>/config.yaml` |
| LanceDB 数据 | `~/.hermes/profiles/<profile>/lance_memory/memories.lance` |
| 原始 session | `~/.hermes/profiles/<profile>/state.db` |
| lancedb-embed 插件 | `~/.hermes/plugins/memory/lancedb-embed/__init__.py` |
| Ollama 服务 | `localhost:11434` |
