#!/usr/bin/env bash
# Task 03 demo: CLI chat + diary memory + feedback + retrieval, backed by our own SFT model.
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
WORK=/root/autodl-tmp/work
cd "$WORK/code/llm" || exit 1
mkdir -p "$WORK/runs/memory"
export ZHCHAT_DB="$WORK/runs/memory/memory.db"
export LLM_BASE_URL="http://127.0.0.1:8010/v1"
export LLM_MODEL="local-sft"
rm -f "$ZHCHAT_DB"

if ! curl -s -m 3 http://127.0.0.1:8010/v1/models > /dev/null; then
  echo "[demo] starting local model server ..."
  (setsid nohup python serve_model.py --ckpt "$WORK/runs/sft/latest.pt" \
     --tokenizer "$WORK/data/processed/tokenizer/zh_bpe.model" --port 8010 \
     > "$WORK/logs/serve.log" 2>&1 < /dev/null &)
  for i in $(seq 1 60); do
    curl -s -m 2 http://127.0.0.1:8010/v1/models > /dev/null && break
    sleep 2
  done
fi
curl -s -m 5 http://127.0.0.1:8010/v1/models; echo

python - <<'PY' | python memory_chat.py --session demo 2>&1 | tee "$WORK/logs/memory_demo.log"
turns = [
 "我叫高旻坤，在做一个小模型预训练项目，代号是「青鸾」，目标是 11 月 25 日前跑完 2 亿 token。",
 "青鸾项目现在卡在数据去重这一步，MinHash 太慢，我改成了按文档抽样 shingle。",
 "帮我记一下：明天上午十点要和同事对一下语料漏斗的统计口径。",
 "我还想加一个功能，把每天的对话压成日记，之后能按关键词检索出来。",
 "确认一下：青鸾项目的截止时间是什么时候？",
 "/diary",
 "/recall 青鸾",
 "/recall 明天",
 "/feedback good 回答简洁，记住了项目代号",
 "/quit",
]
for t in turns:
    print(t)
PY

echo "=== sqlite dump ==="
python - <<'PY'
import os, sqlite3
db = os.environ["ZHCHAT_DB"]
con = sqlite3.connect(db)
print("messages:", con.execute("select count(*) from messages").fetchone()[0])
print("diaries :", con.execute("select count(*) from diaries").fetchone()[0])
print("feedback:", con.execute("select count(*) from feedback").fetchone()[0])
for rid, s, cu in con.execute("select id, summary, covered_until from diaries order by id"):
    print(f"[diary {rid}] covered_until={cu}\n{s}\n")
print("fts rows:", con.execute("select count(*) from diaries_fts").fetchone()[0])
PY
