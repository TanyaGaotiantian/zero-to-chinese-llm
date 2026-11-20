# 01 · 语料工程：把 3 个 dump 变成 112.5M tokens

> 这一章是全书最"脏"的部分，也是最能拉开差距的部分。**预训练模型的质量上限，在这一步就定了。**

## 1. 目标

产出一个可以直接 `np.memmap` 的 token 流，以及一份能解释"每一篇文档为什么被留下/丢掉"的漏斗报告。

| 指标 | 目标 | 本项目 |
|---|---|---|
| 文档数 | 7 万量级 | 71,665 |
| 训练集 tokens | ~112M | **112,503,579** |
| chars/token | 1.5–1.7 | **1.568** |
| 词表 | 32k BPE | 32,000（byte_fallback） |
| 产物 | `train.bin/val.bin/test.bin` | uint16，225 MB / 2.2 MB / 2.4 MB |

## 2. 数据源：先解决"拿得到"

真实的第一道墙不是算法，是带宽。实测（AutoDL 重庆A区，24 并发 range）：

| 通道 | 实测吞吐 | 3.45 GB 需要 |
|---|---:|---|
| `dumps.wikimedia.org`（官方） | **9–80 KB/s**（单连接被限速） | 12 h+ |
| 官方 + 24 路并行 range | ~124 KB/s | ~8 h |
| `dumps.wikimedia.your.org` 镜像（偶发） | 3 MB/s 峰值，随后掉到 124 KB/s | 不可依赖 |
| **HF 镜像的 `wikimedia/wikipedia` parquet** | **5.7 MB/s** | **~2 min/分片** |

结论：**中文维基正文走 parquet，新闻/导游走官方 dump。**

```bash
bash scripts/fetch_sources.sh
# = parquet 2 个分片 (850 MB) + zhwikinews (34 MB) + zhwikivoyage (14 MB)
```

> `wikimedia/wikipedia` 不是"另一个数据集"，它就是官方 dump 的纯文本衍生版（已去 wiki 标记）。为了让三个来源进同一条链，我们把它转成和 `wikiextractor` **完全一样的 JSONL schema**：

```json
{"id": "123", "revid": "", "url": "https://zh.wikipedia.org/wiki?curid=123", "title": "标题", "text": "正文……"}
```

`code/llm/parquet_to_jsonl.py` 只做这一件事，之后的清洗/去重/划分/分词/打包**一行都不用改**。

## 3. 漏斗：485,894 → 71,665

`results/funnel.json` 是真实输出：

| 阶段 | 文档数 | 留存 | 说明 |
|---|---:|---:|---|
| 原始（parquet 461,584 + 新闻/导游） | 485,894 | — | wikiextractor 3.1 输出 |
| 清洗 + 质量闸 | 283,923 | 58.4% | 见下表 |
| 精确去重 + MinHash 近重 | 283,492 | 99.85% | 维基自身重复率很低 |
| 180M 字符预算采样（≥700 字） | **71,665** | 25.3% | 让 tokens 落在 112M |

质量闸的丢弃原因（同 `funnel.json`）：

| 原因 | 数量 | 占比 | 典型样本 |
|---|---:|---:|---|
| `too_short`（<200 字） | 180,799 | 37.2% | 消歧义页、小作品、纯信息框 |
| `low_zh_ratio`（中文 <40%） | 18,457 | 3.8% | 代码块、外文引用、公式页 |
| `repeated_lines` | 1,562 | 0.3% | 模板复制、表格残渣 |
| `too_long`（>20000 字） | 1,098 | 0.2% | 列表页、年鉴 |
| `low_diversity` | 55 | 0.01% | 单字符重复 |

> **为什么要留漏斗**：如果最后 tokens 不够，你需要立刻知道是"源头就少"还是"闸太严"。没有漏斗，只能盲目调参数重跑。

## 4. 七个阶段各自在做什么

### 4.1 抽取（wikiextractor）

```bash
python -m wikiextractor.WikiExtractor data/raw/zhwikinews-...xml.bz2 \
  -o data/interim/extracted/zhwikinews --json --no-templates -q --processes 8
```

- `--json`：输出 JSONL，字段 `id/revid/url/title/text`；
- `--no-templates`：去掉 `{{...}}` 模板（中文维基的信息框几乎全是模板，留着就是噪声）；
- ⚠️ **3.1.0 的输出文件名是 `AA`、`AB`…，不再是 `wiki_00`**。任何 `glob("wiki_*")` 的旧代码会读到 0 篇，且不报错——这是本项目踩过的坑之一。

### 4.2 清洗（`prepare_corpus.py clean`）

逐行规则，顺序很重要：

```python
跳过：空行 / Category:、File:、Template: 等命名空间前缀 / 表格行（{| 、|| 、! ）
标题：== 二级标题 == → 二级标题
去 HTML 标签、[1] 式引用、{{模板}}、[[a|b]] → b、外链、''' 粗体
合并空白
```

然后过质量闸：`min_chars=200`、`max_chars=20000`、`min_zh=0.4`、字符多样性 ≥0.05、重复行比例 <0.7。

### 4.3 去重（精确 + MinHash）

```python
h = sha1(text)                       # 精确重复
m = MinHash(num_perm=128)            # 近似重复（Jaccard ≥ 0.8）
sh = 5-gram 抽样至多 512 个 → m.update_batch(sh)
if lsh.query(m): drop
```

三个细节决定成败：

1. **`cap=512` 的 shingle 抽样**：28 万篇 × 平均 1400 字 ≈ 3 亿个 5-gram，逐条 `update` 要 1 小时以上；确定性抽样后降到 2 分钟，判定质量几乎不变（MinHash 本来就是抽样估计）。
2. **`update_batch` 而不是循环 `update`**：datasketch 内部有向量化路径，快 5–10 倍。
3. **先精确后近似**：精确命中的直接跳过 MinHash，省掉最贵的一步。

### 4.4 预算采样：为什么要"按字符数"卡

目标不是"越多越好"，而是**在固定算力下跑够 epoch 数**。112M tokens ≈ 176M 字符（1.568 字/token），所以直接按字符累加到 180M 就停。

同时加了 `min_chars=700` 的下限：低于 700 字的多数是残页，留着只会拉低平均质量。这一刀把篇数从 123,708 压到 **71,665**，正好落在教程记录的 71,224 附近。

### 4.5 划分（98/1/1）

按**文档**随机划分（不是按行），`seed=42` 保证可复现：

| 划分 | 文档 | tokens |
|---|---:|---:|
| train | 70,233 | 112,503,579 |
| val | 716 | 1,199,254 |
| test | 716 | 1,103,986 |

> val 只有 716 篇够不够？对 125M 模型、20 个 batch 的 eval 来说足够（每 batch 4×1024 token），val_loss 波动 <0.01。要更稳就调 `--val 0.02`。

### 4.6 分词器（BPE 32k）

```python
spm.SentencePieceTrainer.train(
    input="train.txt", vocab_size=32000, model_type="bpe",
    character_coverage=0.9995, byte_fallback=True,
    pad_id=0, unk_id=1, bos_id=-1, eos_id=2, num_threads=12)
```

- `byte_fallback=True`：任何生僻字/emoji 都能回退到字节，**永远不会出 UNK**；
- `character_coverage=0.9995`：覆盖 99.95% 的字符，剩下的走 byte_fallback；
- `bos_id=-1`：文档边界用 EOS 就够，省一个 token；
- 实测 **1.568 字符/token**。如果掉到 1.2 以下，说明分词器没训好（语料太少或 vocab 太大）。

### 4.7 打包（uint16 + EOS）

```python
for doc in docs:
    ids = sp.encode(doc["text"]) + [eos]
    np.asarray(ids, dtype=np.uint16).tofile(f)
```

- 32000 < 65535，`uint16` 够用，**显存/磁盘都省一半**（对比 int32）；
- 文档之间插 EOS，让模型学会"这里该结束"；
- 训练时 `np.memmap` 零拷贝，随机索引任意 `(seq_len+1)` 窗口即可。

## 5. 换成你自己的数据

只要满足"一行一个 JSON，含 `text` 字段"，就能直接进第 4.2 步：

```bash
python code/llm/prepare_corpus.py clean \
  --input my_raw_dir --output data/interim/clean.jsonl --min-chars 300 --min-zh 0.5
python code/llm/prepare_corpus.py dedup --input data/interim/clean.jsonl \
  --output data/interim/dedup.jsonl --threshold 0.85 --max-shingles 512
# … 之后同 scripts/prep2.sh 的 4–8 步
```

常见调整：

| 现象 | 调整 |
|---|---|
| 篇数太少 / tokens 不够 | 降 `min_chars`（700→400）或扩大采样预算 |
| 训练 loss 抖动大 | 提高 `min_chars`、提高 dedup 阈值到 0.85 |
| chars/token < 1.3 | vocab 调小（32k→24k）或增加语料 |
| 显存爆 | 不是语料问题，看 02 章的 `micro-batch` |

## 6. 本章验收

```bash
cat data/processed/stats.json
# train.tokens 应在 110M 量级；chars_per_token 在 1.5–1.7

python - <<'PY'
import numpy as np
a = np.memmap("data/processed/bin/train.bin", dtype=np.uint16, mode="r")
print(len(a), a[:20])          # 长度 = tokens 数，前 20 个 id 应在 3..31999
PY
```

---

下一章：[02 · 预训练](02-pretraining.md)
