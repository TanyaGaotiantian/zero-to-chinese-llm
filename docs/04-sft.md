# 04 · SFT：把 base 变成"会回答问题"的模型

> 预训练教会模型"中文长什么样"，SFT 教会它"该回答哪一句"。这一章的核心只有一件事：**loss 该算在哪几个 token 上。**

## 1. 数据：24,000 条

来源 `YeungNLP/firefly-train-1.1M`（HF 流式读取，不用整包下载）：

```bash
python code/llm/build_sft_data.py --source firefly --n 24000 --out data/sft
# 也可以 --source alpaca_zh / coig
```

过滤规则（`build_sft_data.py:ok()`）：

| 规则 | 理由 |
|---|---|
| `10 ≤ user ≤ 2000` 字 | 太短没信息量，太长超过 seq_len 会被截断 |
| `10 ≤ assistant ≤ 4000` 字 | 同上 |
| `len(assistant) ≥ 0.2 × len(user)` | 剔除"嗯""好的"这类敷衍回答 |
| sha1 去重 | 同一问答反复出现会让 loss 假性下降 |
| 丢弃含"作为一个AI…"的模板回答 | 这类句子会被模型背下来 |

结果：

```json
{"source": "YeungNLP/firefly-train-1.1M", "scanned": 30000, "kept": 24000,
 "train": 23500, "val": 500, "filters": "…"}
```

## 2. 模板与词表扩展

ChatML：

```text
<|im_start|>system
你是一个简洁、诚实的中文助手。不确定时直接说不确定，不要编造。<|im_end|>
<|im_start|>user
中国的首都是哪里？<|im_end|>
<|im_start|>assistant
北京。<|im_end|>
```

**词表扩展**：base 的 spm 词表是 32,000（id 0..31999），SFT 追加两个特殊 token：

```python
IM_START, IM_END = 32000, 32001
cfg = GPTConfig(vocab_size=32000 + 2, ...)     # embedding 多两行
```

加载 base 权重时只拷贝前 32,000 行，新加的两行随机初始化：

```python
if k == "tok_emb.weight":
    msd[k][: v.shape[0]].copy_(v)      # 前 32000 行来自 base
```

> 为什么不用 base 词表里现成的 `<s>`/`</s>`？因为 spm 的 `<s>` 是 BOS，语义不同；专用 token 能让模型学出"轮次边界"，对多轮对话很关键。

## 3. 关键：assistant-only loss

```python
def _encode(self, messages):
    ids, labels = [], []
    for m in messages:
        ids.append(IM_START);            labels.append(-100)
        prefix = sp.encode(f"{m['role']}\n")
        ids += prefix;                   labels += [-100] * len(prefix)
        cids = sp.encode(m["content"])
        ids += cids
        labels += (cids if m["role"] == "assistant" else [-100] * len(cids))
        ids.append(IM_END);              labels.append(IM_END if m["role"] == "assistant" else -100)
    return ids[:self.max_len], labels[:self.max_len]
```

只有 **assistant 的内容 + 结尾的 `<|im_end|>`** 参与梯度，其余全是 -100。

为什么必须这样：

| 做法 | val_loss | 生成质量 |
|---|---:|---|
| 全序列都算 loss | 更低（虚低） | 差：模型学会"复述问题"而不回答 |
| **只算 assistant**（本项目） | 更高（真实） | 好：梯度全部用于"怎么答" |

> 自检方法：把一条样本喂进去，打印 `labels` 里非 -100 的数量，应该 ≈ assistant 的 token 数（+1 个 im_end），而不是整条长度。

batch 内 padding 也用 -100：

```python
y = torch.full((B, maxlen), -100, dtype=torch.long)   # 先全 -100，再填真实 label
```

## 4. 超参

| 项 | 值 | 说明 |
|---|---|---|
| init | `runs/125m/latest.pt` | 从 base 续，不是从头 |
| seq_len | 1024 | 超长样本截断（很少见） |
| batch × accum | 8 × 4 = 32 条/步 | 约 1500 步 / 2 epoch |
| lr | 1e-4（cosine → 10%） | 实测 1e-5 收敛太慢（终值 4.42），1e-4 明显更好 |
| weight decay | 0.01 | 只对 ≥2 维 |
| warmup | 50 | |
| epochs | 3 | 24k 数据；第 3 轮末段开始过拟合（见下表） |

## 5. 结果

```text
[init] loaded runs/125m/latest.pt (new special-token rows are randomly initialised)
[data] train=23500 val=500 steps/epoch=735 total=2205

[eval] step  100 val_loss 4.8698 ppl 130.30
[eval] step  400 val_loss 4.3607 ppl  78.31
[eval] step  800 val_loss 4.3902 ppl  80.37
[eval] step 1200 val_loss 4.6823 ppl 108.01
[eval] step 1600 val_loss 4.2904 ppl  73.02
[eval] step 1900 val_loss 4.0039 ppl  54.81
[eval] step 2000 val_loss 3.8069 ppl  45.01   ← 最好
[eval] step 2200 val_loss 4.1954 ppl  66.38
[done] 2205
```

**三次 SFT 的对照**：

| 运行 | 配置 | 终值 val_loss | 结论 |
|---|---|---:|---|
| 错位 bug 版 | lr 1e-5 / 2 epoch / **没做 shift** | **0.0005** | 退化解：学到"复制输入"，loss 假性归零，生成完全不可用（`results/sft_buggy.log`） |
| 低学习率版 | lr 1e-5 / 2 epoch | 4.4161 | 收敛太慢 |
| **正式版** | lr 1e-4 / 3 epoch | **3.8069**（最好）/ 4.1954（末） | 采用 |

**为什么没到 2.4？** 这是本章最诚实的一段：

- 125M 参数只吃了 **112.5M tokens**（≈1.9 token/参数），而 Chinchilla 式的计算最优大约是 **20 token/参数**——也就是说 base 本身欠训约 10 倍，它的 wiki val_loss 就是 4.54；
- SFT 只能在这个 base 上"对齐格式 + 压缩答案分布"，无法凭空补回预训练欠下的语言能力；
- 要让 SFT val_loss 进 2.4 区间，实际需要：base 再吃 10–20 倍 tokens（¥40–80 的算力），或换 360M/1B 规模。

**这恰恰说明了 SFT 的边界**：SFT 不会让模型变聪明，它只让模型"按你要的格式输出它已经会的东西"。这个结论比一个漂亮的 2.4 更有价值。

> 如果 val_loss 掉到 0.01 以下，**先怀疑是错位 bug**（见 §3 的 `pairs = [(ids[:-1], labels[1:])]`），而不是庆祝。

## 6. 把模型变成接口

不需要 vLLM：`serve_model.py` 用 `http.server` 起一个 OpenAI 兼容端点。

```bash
python code/llm/serve_model.py --ckpt runs/sft/latest.pt \
       --tokenizer data/processed/tokenizer/zh_bpe.model --port 8010
curl -s localhost:8010/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"local-sft",
  "messages":[{"role":"user","content":"用一句话解释什么是梯度"}],
  "max_tokens":80}' | python -m json.tool
```

它做的事：拼 ChatML → 生成 → 在第一个 `<|im_end|>` 截断 → 返回 OpenAI 格式。**没有 KV cache**，200 token 大约 1–2 秒（125M 模型），够做原型。

## 7. 三个常见坑

| 坑 | 现象 | 修法 |
|---|---|---|
| 特殊 token 越界 | `IndexError: index out of range in self` | `vocab_size` 必须 = 32000+2，且 embedding 多做两行 |
| padding 参与 loss | val_loss 异常低，生成全是重复 | `y` 初始化为 -100，只在真实位置填 |
| 从 base 加载时 shape 不匹配 | `size mismatch for tok_emb.weight` | 手动拷前 N 行（见 §2），不要直接 `load_state_dict` |

## 8. 本章验收

```bash
grep "\[eval\]" logs/sft.log | tail -3      # 最后一条 val_loss < 2.6
python - <<'PY'
import json
rows=[json.loads(l) for l in open("data/sft/train.jsonl")]
print(len(rows), rows[0]["messages"][1]["role"], rows[0]["messages"][1]["content"][:30])
PY
```

---

上一章：[03 · 断点续训](03-resume.md)　下一章：[05 · 记忆原型](05-memory.md)
