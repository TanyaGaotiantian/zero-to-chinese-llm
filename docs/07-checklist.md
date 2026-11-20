# 07 · 复现检查清单

> 打印这一页，边做边勾。**每一步都有"期望输出"，对不上就别往下走。**

## 阶段 0 · 环境

- [ ] GPU 能看到：`nvidia-smi` 显示 24 GB 卡
- [ ] `python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability())"`
      期望 `2.8.x (8, 9)`（4090）或 `(12, 0)`（5090）
- [ ] `df -h /root/autodl-tmp` 可用 ≥ 40 GB
- [ ] 数据目录已建：`mkdir -p /root/autodl-tmp/work && cd $_`

## 阶段 1 · 取数

- [ ] `bash scripts/fetch_sources.sh` 结束打印 `SOURCES_DONE`
- [ ] `ls -l data/raw` 里 parquet 是 **587,109,027** 字节（分片 0）
- [ ] `bzip2 -t data/raw/zhwikinews-*.bz2` 无报错
- [ ] 总量：`du -sh data/raw` ≈ 650 MB

> 如果维基 dump 下不动（<100 KB/s），别硬等，直接用 parquet 通道。

## 阶段 2 · 语料

- [ ] `bash scripts/prep2.sh` 结束打印 `=== PREP DONE ===`
- [ ] `cat data/processed/stats.json`：`train.tokens` 在 110M 量级、`chars_per_token` 1.5–1.7
- [ ] `cat data/interim/funnel.json`：`raw` ≈ 485,000、`kept_ratio` ≈ 0.58
- [ ] `ls -l data/processed/bin`：`train.bin` ≈ 225 MB
- [ ] `python -c "import numpy as np;a=np.memmap('data/processed/bin/train.bin',dtype=np.uint16,mode='r');print(len(a),a[:8])"`
      期望长度 > 1e8，前 8 个 id 在 3..31999

## 阶段 3 · 冒烟

- [ ] `bash scripts/stages.sh smoke`
- [ ] 日志里 `step 1 | loss 10.5` 左右（≈ ln(32000)=10.37）
- [ ] 20 步后 `loss < 9.0`
- [ ] `[mem] peak allocated` 在 11–13 GiB（125M/16×1024）
- [ ] 吞吐 > 80k tok/s（4090 级）

> 冒烟不过就别开长跑——这一步 2 分钟，能省 2 小时。

## 阶段 4 · 续训

- [ ] `bash scripts/stages.sh resume`
- [ ] `logs/verify_resume.txt` 最后一行是 `PASS`
- [ ] `max |loss diff| < 0.05`（本项目 0.0100）
- [ ] `batch fingerprints identical on all overlapping steps`
- [ ] `logs/resume_killed.log` 里能看到 `Killed`（确认真的是 SIGKILL）

## 阶段 5 · 预训练

- [ ] `bash scripts/stages.sh 125m`，50±10 min
- [ ] `tail -1 runs/125m/train.csv` 的 step = 699
- [ ] `grep "\[eval\]" logs/125m.log` 最后一条 `val_loss < 4.8`
- [ ] `grep "\[mem\]" logs/125m.log` allocated ≈ 11.15 GiB
- [ ] `cat runs/125m/samples.txt`：中文通顺（不必正确）

- [ ] `bash scripts/stages.sh 360m`，2±0.3 h
- [ ] `tail -1 runs/360m/train.csv` 的 step = 456
- [ ] `val_loss < 5.6`

## 阶段 6 · SFT

- [ ] `bash scripts/stages.sh sft`
- [ ] `cat data/sft/card.json`：`kept` = 24000
- [ ] `grep "\[eval\]" logs/sft.log` 最后一条 `val_loss < 2.6`
- [ ] `ls -l runs/sft/latest.pt` 存在

## 阶段 7 · 记忆原型

- [ ] `bash scripts/demo_memory.sh`
- [ ] `logs/memory_demo.log` 里出现 `[已写入日记]`
- [ ] `/recall llm-lab` 能返回包含“llm-lab”的日记
- [ ] sqlite 里三张表都有数据：

```bash
python - <<'PY'
import sqlite3, os
con = sqlite3.connect(os.path.expanduser("~/.zhchat/memory.db"))
for t in ("messages","diaries","feedback"):
    print(t, con.execute(f"select count(*) from {t}").fetchone()[0])
PY
```

## 阶段 8 · 收尾（别忘！）

- [ ] 把 `runs/*/train.csv`、`logs/*.log`、`data/processed/stats.json` 拷出来
- [ ] `python code/llm/export_model.py runs/125m/latest.pt export/125m_model.pt`（去掉优化器，体积减 2/3）
- [ ] **关机**：控制台 → 容器实例 → 关机（GPU 计费立即停止）
- [ ] 确认控制台余额不再下降（隔 5 分钟刷一次）

> 最后一条最容易忘。**空转 1 小时 = ¥1.88 = 又跑了一次 125M。**

## 常见"对不上"速查

| 现象 | 最可能的原因 | 去哪看 |
|---|---|---|
| `funnel.json` 里 `raw=0` | wikiextractor 3.1 输出名变了 | [01 §4.1](01-corpus.md) |
| `chars_per_token < 1.3` | 分词器没训好 / vocab 太大 | [01 §4.6](01-corpus.md) |
| 续训 `batch_sha1` 不一致 | 数据顺序没存 / numpy 版本变了 | [03 §2](03-resume.md) |
| SFT loss 一直不降 | label 全 -100（mask 写错） | [04 §3](04-sft.md) |
| OOM at 360M | 没开 grad checkpoint | [02 §2](02-pretraining.md) |
| 内存/吞吐骤降 | `torch.compile` 撞上动态 shape | [02 §5](02-pretraining.md) |

---

上一章：[06 · 成本与踩坑](06-cost-and-pitfalls.md)
