"""A local-first chat CLI with diary-style long-term memory.

Works with any OpenAI-compatible endpoint (Ollama / vLLM / llama.cpp server):
  export LLM_BASE_URL=http://127.0.0.1:11434/v1
  export LLM_MODEL=qwen2.5:7b-instruct
  python memory_chat.py --session work

Storage: ~/.zhchat/memory.db (SQLite + FTS5 trigram). No network calls except
to the endpoint you configure.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

DB_PATH = Path(os.environ.get("ZHCHAT_DB", Path.home() / ".zhchat" / "memory.db"))
BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b-instruct")
API_KEY = os.environ.get("LLM_API_KEY", "local")


def db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY, session TEXT, role TEXT, content TEXT, ts REAL);
    CREATE TABLE IF NOT EXISTS diaries(
        id INTEGER PRIMARY KEY, session TEXT, ts REAL, summary TEXT, covered_until INTEGER);
    CREATE VIRTUAL TABLE IF NOT EXISTS diaries_fts USING fts5(
        summary, content='diaries', content_rowid='id', tokenize='trigram');
    CREATE TABLE IF NOT EXISTS feedback(
        id INTEGER PRIMARY KEY, message_id INTEGER, score INTEGER, note TEXT, ts REAL);
    """)
    return con


def chat(messages, temperature=0.7, max_tokens=800):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


def add_msg(con, session, role, content):
    cur = con.execute("INSERT INTO messages(session,role,content,ts) VALUES(?,?,?,?)",
                      (session, role, content, time.time()))
    con.commit()
    return cur.lastrowid


def recent(con, session, limit=12):
    rows = con.execute(
        "SELECT role,content FROM messages WHERE session=? ORDER BY id DESC LIMIT ?",
        (session, limit)).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def add_diary(con, session, summary, covered_until):
    cur = con.execute("INSERT INTO diaries(session,ts,summary,covered_until) VALUES(?,?,?,?)",
                      (session, time.time(), summary, covered_until))
    con.execute("INSERT INTO diaries_fts(rowid, summary) VALUES(?,?)", (cur.lastrowid, summary))
    con.commit()


def recall(con, session, query, k=3):
    q = query.strip()
    if len(q) < 3:
        rows = con.execute(
            "SELECT summary FROM diaries WHERE session=? ORDER BY id DESC LIMIT ?",
            (session, k)).fetchall()
    else:
        try:
            rows = con.execute(
                "SELECT d.summary FROM diaries_fts f JOIN diaries d ON d.id=f.rowid "
                "WHERE diaries_fts MATCH ? AND d.session=? ORDER BY rank LIMIT ?",
                (q, session, k)).fetchall()
        except sqlite3.OperationalError:
            rows = con.execute(
                "SELECT summary FROM diaries WHERE session=? AND summary LIKE ? "
                "ORDER BY id DESC LIMIT ?", (session, f"%{q}%", k)).fetchall()
    return [r[0] for r in rows]


def summarise(con, session, keep_recent=6):
    rows = con.execute(
        "SELECT id,role,content FROM messages WHERE session=? ORDER BY id", (session,)).fetchall()
    if len(rows) <= keep_recent + 6:
        return None
    last_covered = con.execute(
        "SELECT COALESCE(MAX(covered_until),0) FROM diaries WHERE session=?", (session,)).fetchone()[0]
    todo = [r for r in rows if r[0] > last_covered][:-keep_recent]
    if len(todo) < 8:
        return None
    transcript = "\n".join(f"{r[1]}: {r[2]}" for r in todo)[-6000:]
    prompt = (
        "把下面的对话压缩成一篇日记，用第三人称，200 字以内。必须包含："
        "日期/时间段、涉及的人或项目、确认的事实、做出的决定、未完成的事项。"
        "不要编造，不要写评价。\n\n" + transcript)
    summary = chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=300)
    add_diary(con, session, summary, todo[-1][0])
    return summary


def build_system(con, session, user_text):
    mem = recall(con, session, user_text)
    base = ("你是一个本地运行的诚实助手。回答简洁、具体；不确定就说不确定。"
            "不要声称自己访问了网络。")
    if mem:
        base += "\n\n【长期记忆（日记摘录，可能不完整）】\n" + "\n---\n".join(mem)
    return base


def repl(session):
    con = db()
    print(f"session={session} db={DB_PATH} model={MODEL}")
    print("命令：/diary  /recall 词  /feedback good|bad [note]  /quit")
    while True:
        try:
            text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/diary":
            s = summarise(con, session)
            print("[diary]", s or "(对话还不够长，未生成)")
            continue
        if text.startswith("/recall"):
            q = text[len("/recall"):].strip()
            for i, s in enumerate(recall(con, session, q, k=5)):
                print(f"[{i}] {s}\n")
            continue
        if text.startswith("/feedback"):
            parts = text.split(maxsplit=2)
            score = 1 if len(parts) > 1 and parts[1] == "good" else -1
            mid = con.execute("SELECT MAX(id) FROM messages WHERE session=?", (session,)).fetchone()[0]
            con.execute("INSERT INTO feedback(message_id,score,note,ts) VALUES(?,?,?,?)",
                        (mid, score, parts[2] if len(parts) > 2 else "", time.time()))
            con.commit()
            print("[feedback] saved")
            continue

        add_msg(con, session, "user", text)
        msgs = [{"role": "system", "content": build_system(con, session, text)}] + recent(con, session)
        try:
            answer = chat(msgs)
        except Exception as e:
            print(f"[error] {e}\n检查 LLM_BASE_URL / 模型是否已启动。", file=sys.stderr)
            continue
        print("\n助手>", answer)
        add_msg(con, session, "assistant", answer)
        s = summarise(con, session)
        if s:
            print("\n[已写入日记]", s[:80], "...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="default")
    args = ap.parse_args()
    repl(args.session)
