# 直接跑这个模型

三种方式，从"什么都不装"到"本地全量"：

| 方式 | 需要什么 | 命令 |
|---|---|---|
| **① 看现成输出** | 什么都不用 | 打开 [samples.md](samples.md)（真实输出，未挑选） |
| **② Colab 一键** | 一个浏览器 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TanyaGaotiantian/zero-to-chinese-llm/blob/main/demo/colab.ipynb) |
| **③ 本地 Web UI** | Python + 约 300 MB 权重 | `pip install gradio sentencepiece torch` → `python demo/app.py` |

第一次运行会自动从 [release v1.0-models](https://github.com/TanyaGaotiantian/zero-to-chinese-llm/releases/tag/v1.0-models)
下载 `gpt125m-sft-bf16.pt`（约 287 MiB）和 `zh_bpe.model`，缓存到 `~/.cache/zero-to-chinese-llm/`。
也可以用 `ZHLLM_CACHE=/your/dir` 改缓存位置。

## 本地 Web UI

```bash
pip install torch sentencepiece gradio
python demo/app.py                 # http://127.0.0.1:7860
python demo/app.py --share         # 生成一个临时公网链接
python demo/app.py --port 8888
```

界面里有三个旋钮，值得玩一下它们对输出的影响：

| 参数 | 作用 | 建议 |
|---|---|---|
| `temperature` | 越高越发散 | 0.6–0.9；>1.2 开始胡言乱语 |
| `top-p` | 只从累积概率前 p 里采样 | 0.85–0.95 |
| `repetition penalty` | 压制最近 64 个 token | **1.2–1.4；1.0 会立刻陷入复读** |

## 命令行

```bash
python demo/app.py --once "用一句话解释什么是梯度"   # 单次生成
python demo/app.py --cli                            # 交互式 REPL
```

## 用自己训练的 ckpt

```bash
python demo/app.py --ckpt runs/sft/latest.pt --tokenizer data/processed/tokenizer/zh_bpe.model
```

`app.py` 只依赖 `code/llm/model.py` 里那 230 行模型定义，**不依赖本项目其他任何东西**——可以单独拷走。

## 为什么不做成纯浏览器 WASM demo

125M 参数的 fp16 是 250 MB，q8 量化后仍有 ~130 MB，首屏要下载几百 MB、CPU 推理还要几百毫秒/token。
相比 Colab（免费 GPU + 一条链接），体验更差、维护成本更高。**能用一行链接解决的事，不引入 WASM 工具链。**

## 想部署成常驻 Demo

1. 在 Hugging Face 建一个 **Space**（Gradio 模板）；
2. 把 `demo/app.py` 和 `code/llm/model.py` 拖进去，`requirements.txt` 写 `torch sentencepiece gradio`；
3. 权重放 Space 的 Files 或继续用本仓库 release 的直链。

`app.py` 会自动下载权重，所以 Space 上只需要几十行配置。
