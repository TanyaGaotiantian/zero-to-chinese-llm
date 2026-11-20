# 03 · 断点续训的硬验证：kill -9 之后，训练还能"接上"吗

> 这是全书**最容易被糊弄过去**的一章。大多数项目说"支持续训"，实际只恢复了模型权重——loss 曲线会在恢复点出现一个台阶，而没人去比对。

## 1. 为什么续训很难

一次训练 step 的"状态"远不止模型权重：

| 状态 | 不恢复会怎样 |
|---|---|
| 模型参数 | 直接从头开始 |
| 优化器状态（AdamW 的 exp_avg / exp_avg_sq） | 前几十步 loss 突跳（动量被清空） |
| 全局 step | 学习率调度错位（cosine 从头算） |
| tokens_seen | 吞吐/预算统计错 |
| **数据顺序**（epoch + pos + 置换） | 恢复后从 epoch 0 开头读，**重复训练同一批数据** |
| **CPU/CUDA RNG** | dropout/采样不一致，loss 不可复现 |

**最难的是数据顺序**。如果 dataloader 用 `shuffle=True`，随机状态没有显式保存，恢复后顺序必然不同——这一项错了，别的都对也没用。

## 2. 本项目的做法

### 2.1 确定性置换，而不是"随机打乱"

```python
class BinDataset:
    def __init__(...):
        self.order = np.random.default_rng(seed).permutation(self.n)
    def _new_epoch(self):
        self.epoch += 1
        self.order = np.random.default_rng(seed + self.epoch).permutation(self.n)
        self.pos = 0
```

顺序是 `(seed, epoch)` 的**纯函数**，不依赖任何隐藏状态。ckpt 里存：

```python
"data": {"epoch": 3, "pos": 12800, "n": 343572, "order_sha1": "9f2c…"}
```

恢复时重建置换并**校验 sha1**：

```python
self.order = np.random.default_rng(seed + st["epoch"]).permutation(self.n)
assert sha1(self.order) == st["order_sha1"], "data order mismatch: numpy version changed?"
self.pos = st["pos"]
```

> sha1 校验那行不是装饰：换机器、换 numpy 版本、换 `n`，都会让置换变化。**宁可启动就报错，也不要静默地训错。**

### 2.2 ckpt 里到底存了什么

```python
ckpt = {
    "model":  unwrap(model).state_dict(),
    "optim":  optim.state_dict(),
    "step":   step,
    "tokens": tokens_seen,
    "config": asdict(cfg),
    "data":   ds.state_dict(),            # epoch/pos/n/sha1
    "args":   vars(args),
    "torch_rng": torch.get_rng_state(),
    "cuda_rng":  torch.cuda.get_rng_state_all(),
}
```

### 2.3 CSV 的"截断"细节

崩溃时，`train.csv` 可能已经写到 step 26，但最后一个 ckpt 是 step 20。续训如果直接 append，就会出现 21–26 的**重复行**，`verify_resume.py` 比对时永远对不上。所以恢复时要先截断：

```python
if resume_path and start_step > 0:
    keep = [rows[0]] + [r for r in rows[1:] if int(r[0]) <= start_step]
```

## 3. 验证协议（可复制）

**A 组（基线）**：不间断跑 60 步。
**B 组（中断）**：同配置跑，约 step 25 时 `kill -9`（SIGKILL，不给清理机会），再从 `latest.pt`（step 20）续到 60 步。
**比对**：逐 step 比 `loss` 与 `batch_sha1`。

```bash
bash scripts/stages.sh resume
```

核心就三行：

```bash
timeout -s KILL 100 python train.py ... --run-dir runs/resume_test --max-steps 60   # B 组被杀
python train.py ... --run-dir runs/resume_test --max-steps 60                       # 自动 resume
python verify_resume.py runs/resume_ref/train.csv runs/resume_test/train.csv --tol 0.05
```

## 4. 实测结果

```text
overlap steps: 60  range: 1..60
max |loss diff| = 0.009987  mean = 0.002670
batch fingerprints identical on all overlapping steps
PASS
```

| 指标 | 结果 | 说明 |
|---|---|---|
| 重叠步数 | 60 / 60 | 全覆盖 |
| batch 指纹（`batch_sha1`） | **60/60 完全一致** | 数据顺序被精确恢复 |
| max abs(Δloss) | **0.009987** | 远小于 0.05 容差 |
| mean abs(Δloss) | 0.002670 | 差异来自 bf16 累加的非确定性 |

`batch_sha1` 是每步第一个 micro-batch 输入的 sha1，它一致 = **喂给模型的数据一模一样**；loss 一致 = 优化器/RNG 状态也回来了。两个都对，才叫"续训"。

> 为什么 loss 不是 0 差？因为 bf16 的矩阵乘累加顺序在编译优化后会变，这是**硬件/编译器层面的非确定性**，不是状态丢失。A 组自己跑两遍也会有 1e-3 量级的差异。

## 5. 五个常见错误

| 错误 | 现象 | 修法 |
|---|---|---|
| 只存 `model.state_dict()` | 恢复后 loss 跳升 0.3+，几十步才回落 | 必须存 `optim.state_dict()` |
| dataloader 用 `shuffle=True` | 恢复后 loss 正常但**重复见到旧数据**（看不出来） | 用 `(seed, epoch)` 生成置换并校验 sha1 |
| 学习率按"本次启动的 step"算 | 曲线在恢复点出现折角 | step 要存全局值，`lr_at(global_step, max_steps, …)` |
| 不存 RNG | 结果不可复现，A/B 比对永远失败 | 存 `torch_rng` + `cuda_rng` |
| CSV append 不截断 | 出现重复 step 行 | 恢复时按 `start_step` 截断 |

## 6. 本章验收

```bash
bash scripts/stages.sh resume
# 期望最后三行：
#   max |loss diff| = 0.0xxxxx
#   batch fingerprints identical on all overlapping steps
#   PASS

# 想更严：把容差压到 0.02
python code/llm/verify_resume.py runs/resume_ref/train.csv runs/resume_test/train.csv --tol 0.02
```

---

上一章：[02 · 预训练](02-pretraining.md)　下一章：[04 · SFT](04-sft.md)
