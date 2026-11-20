# 05 · 记忆原型：把历史对话压成日记，还要能检索回来

> 大模型的上下文是有限的，人的记忆也不是逐字存储的。这一章做一个**最小但真实**的长期记忆：
> 对话超过阈值 → 压成日记 → 存 SQLite → 之后按关键词检索回注入。

## 1. 需求拆成四件事

| 需求 | 实现 |
|---|---|
| 对话要能存下来 | `messages` 表（session 隔离） |
| 历史要压缩 | 日记式摘要，记录 `covered_until` 防止重复压缩 |
| 之后要能找到 | FTS5 trigram 索引，支持中文子串检索 |
| 用户要能反馈 | `feedback` 表，good/bad + 备注，挂在具体消息上 |

## 2. 表结构

```sql
CREATE TABLE messages(id INTEGER PRIMARY KEY, session TEXT, role TEXT, content TEXT, ts REAL);
CREATE TABLE diaries(id INTEGER PRIMARY KEY, session TEXT, ts REAL, summary TEXT, covered_until INTEGER);
CREATE VIRTUAL TABLE diaries_fts USING fts5(summary, content='diaries', content_rowid='id', tokenize='trigram');
CREATE TABLE feedback(id INTEGER PRIMARY KEY, message_id INTEGER, score INTEGER, note TEXT, ts REAL);
```

三个设计点：

1. **`covered_until`**：记录"已经压缩到哪条消息"，下次只压缩新增部分。没有它就会反复压缩同一段历史，日记越写越重复。
2. **`tokenize='trigram'`**：SQLite 默认分词器对中文无效（整段当一个词）。trigram 把文本切成三字滑窗，像「llm-lab」这样的专有名词也能命中。
3. **`content='diaries'`（external content）**：FTS 表不存正文副本，只存索引，省空间且不会与主表不一致。

## 3. 压缩策略（这里有个真坑）

```python
if len(rows) <= keep_recent + 6: return None          # 太短不压
todo = [r for r in rows if r[0] > last_covered][:-keep_recent]   # 保留最近 6 条
if len(todo) < 8: return None
transcript = "\n".join(f"{r[1]}: {r[2]}" for r in todo)[-6000:]
```

然后交给模型压成 200 字以内的第三人称日记。

**真坑**：我们手头的模型只有 125M，它写出来的"日记"是这样的：

```text
1 人们都知道,在哪里? 3 我叫高芷若! 4 我叫高芷若! 5 我叫高芷若! …
```

这不是代码 bug，是**模型能力问题**。如果直接入库，`/recall` 检索到的全是垃圾，记忆系统等于没有。

所以在工程上做了两个决定：

| 决定 | 理由 |
|---|---|
| **日记默认用抽取式摘要**（`_digest()`） | 只保留"用户真正说过的话"，不生成、不编造；关键事实一定在 |
| **模型复述作为可选增强**（`ZHCHAT_LLM_DIARY=1`） | 换成本地 Qwen2.5-7B / 任意强模型时再打开，用 `_usable()` 质量闸拦掉退化输出 |

质量闸（`_usable`）检查：长度 ≥40、无 6 连重复字符、3-gram 最高频占比 ≤5%、4-gram ≤4%。这四条能拦住几乎所有"复读机"输出。

> 这是本章最值得抄的一段：**记忆系统不能把"模型输出"直接当事实存起来**。要么用抽取式兜底，要么加质量闸。

## 4. 检索与注入

```python
def build_system(con, session, user_text):
    mem = recall(con, session, user_text)          # 用当前问题去检索日记
    base = "你是一个本地运行的诚实助手……"
    if mem:
        base += "\n\n【长期记忆（日记摘录，可能不完整）】\n" + "\n---\n".join(mem)
    return base
```

每轮都做一次检索，把命中的日记拼进 system prompt。注意措辞"**可能不完整**"——这是在提示模型不要把这部分当全知事实。

## 5. 实测（125M SFT 模型当后端）

```bash
bash scripts/demo_memory.sh   # 起 serve_model + 跑一段 8 轮对话 + dump sqlite
```

结果：

```text
messages 16
diaries 1
feedback 1
[diary 1] 2026-11-20 对话摘要（抽取式）：我叫高旻坤，在做一个小模型预训练项目，代号是「llm-lab」，
目标是本周五前跑完 2 亿 token。；llm-lab 项目现在卡在数据去重这一步，MinHash 太慢，我改成了按文档
抽样 shingle。；帮我记一下：明天上午十点要和同事对一下语料漏斗的统计口径。；另外 125M 那个模型
的峰值显存是 11.15 GiB，比我预期低。
fts_rows 1
```

- `/recall llm-lab` 命中这条日记（FTS5 trigram）；
- `/feedback good 回答了项目代号` 写入 `feedback`；
- 重启进程后数据仍在（SQLite 文件），这就是"本地持久化"。

生成的对话内容本身很差（125M 模型），但**记忆的存取、压缩、检索、反馈四条链路都是通的**——换成任何 OpenAI 兼容端点即可直接用于真实场景。

## 6. 升级路径

| 现在 | 下一步 |
|---|---|
| 抽取式日记 | `ZHCHAT_LLM_DIARY=1` + Qwen2.5-7B（Ollama）→ 真正的日记 |
| FTS5 关键词 | 混合检索：FTS5 召回 + embedding 重排（接口不变） |
| 单文件 SQLite | 多用户时换 Postgres + pgvector |
| 只注入 system prompt | 加入"记忆引用"返回，让用户看到答案依据哪条日记 |

## 7. 本章验收

```bash
export LLM_BASE_URL=http://127.0.0.1:8011/v1 LLM_MODEL=local-sft
python code/llm/memory_chat.py --session demo
# 输入 8 轮对话后：
/diary           # 应输出一条包含关键事实的摘要
/recall llm-lab      # 应命中
/feedback good ok
```

```bash
python - <<'PY'
import sqlite3, os
con = sqlite3.connect(os.path.expanduser("~/.zhchat/memory.db"))
print(con.execute("select count(*) from messages").fetchone(),
      con.execute("select count(*) from diaries").fetchone(),
      con.execute("select count(*) from feedback").fetchone())
PY
```

---

上一章：[04 · SFT](04-sft.md)　下一章：[06 · 成本与踩坑](06-cost-and-pitfalls.md)
