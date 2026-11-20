# 05 · Memory prototype: compress history into diaries, then retrieve it

> Context windows are finite and human memory is not verbatim. This chapter builds a **minimal but real**
> long-term memory: compress old turns into diary entries, persist them in SQLite, retrieve by keyword later.

## 1. Four requirements

| Requirement | Implementation |
|---|---|
| Persist conversation | `messages` table (per session) |
| Compress history | diary summaries + `covered_until` so nothing is compressed twice |
| Find it later | FTS5 `trigram` index (works for Chinese substrings) |
| Collect feedback | `feedback` table, good/bad + note, attached to a message |

## 2. Schema

```sql
CREATE TABLE messages(id INTEGER PRIMARY KEY, session TEXT, role TEXT, content TEXT, ts REAL);
CREATE TABLE diaries(id INTEGER PRIMARY KEY, session TEXT, ts REAL, summary TEXT, covered_until INTEGER);
CREATE VIRTUAL TABLE diaries_fts USING fts5(summary, content='diaries', content_rowid='id', tokenize='trigram');
CREATE TABLE feedback(id INTEGER PRIMARY KEY, message_id INTEGER, score INTEGER, note TEXT, ts REAL);
```

1. **`covered_until`** — the id of the last message already compressed; without it the same
   history is re-summarised forever.
2. **`trigram`** — the default tokenizer treats a whole Chinese sentence as one token; trigram
   lets “llm-lab” match.
3. **external content FTS** — index only, no duplicate body, no drift.

## 3. Compression (and the trap)

```python
if len(rows) <= keep_recent + 6: return None
todo = [r for r in rows if r[0] > last_covered][:-keep_recent]
if len(todo) < 8: return None
```

Then ask the model for a 200-character third-person diary. **With the 125M model you get this:**

```text
1 人们都知道,在哪里? 3 我叫高芷若! 4 我叫高芷若! 5 我叫高芷若! …
```

Not a bug — a capability limit. Storing that would poison `/recall`. Hence two engineering decisions:

| Decision | Why |
|---|---|
| **Diaries are extractive by default** (`_digest()`) | keeps what the user actually said; never invents |
| **Model narrative is opt-in** (`ZHCHAT_LLM_DIARY=1`) | enable with a real model (Qwen2.5-7B via Ollama), guarded by `_usable()` |

`_usable()` rejects: length < 40, six repeated characters, a 3-gram appearing in > 5% of windows,
a 4-gram in > 4%. That kills nearly every degenerate loop.

> The transferable lesson: **never store raw model output as a fact in a memory system.**
> Either fall back to extraction or gate it.

## 4. Retrieval and injection

```python
mem = recall(con, session, user_text)     # query with the current question
if mem:
    base += "\n\n【长期记忆（日记摘录，可能不完整）】\n" + "\n---\n".join(mem)
```

The "可能不完整" wording is deliberate: it tells the model not to treat memory as ground truth.

## 5. Measured demo (125M SFT backend)

```text
messages 16
diaries 1
feedback 1
[diary 1] 2026-11-20 对话摘要（抽取式）：我叫高旻坤，在做一个小模型预训练项目，代号是「llm-lab」…
fts_rows 1
```

`/recall llm-lab` hits the diary via FTS5; `/feedback good …` is stored; the SQLite file survives restarts.
The chat replies themselves are poor (125M model) but the four links — store, compress, retrieve,
feedback — all work, and swapping in any OpenAI-compatible endpoint makes it usable.

## 6. Upgrade path

| Now | Next |
|---|---|
| extractive diary | `ZHCHAT_LLM_DIARY=1` + Qwen2.5-7B for real diaries |
| FTS5 keywords | hybrid retrieval (FTS5 recall + embedding rerank), same interface |
| single SQLite file | Postgres + pgvector for multi-user |
| injected into system prompt | return "memory citations" so users see the evidence |

## 7. Verify

```bash
export LLM_BASE_URL=http://127.0.0.1:8011/v1 LLM_MODEL=local-sft
python code/llm/memory_chat.py --session demo
```

---

Previous: [04 · SFT](04-sft.md) · Next: [06 · Cost and pitfalls](06-cost-and-pitfalls.md)
