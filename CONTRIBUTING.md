# Contributing

Thanks for looking! This repository is a **reproducibility report**, so the bar for changes is:

1. **Numbers must be measured.** If you change a hyperparameter, include the new `train.csv`
   row or log line in the PR description.
2. **Every stage stays idempotent.** Re-running a script must not redo finished work.
3. **No new heavyweight dependencies.** The whole point is `torch + numpy + sentencepiece`.
4. **Both languages.** Docs live in `docs/` (zh) and `docs/en/`; keep them in sync.

## Good first issues

- Add a `--rope-scale` option and a 4K-context comparison table.
- Add a Chinese eval subset (CEval/CMRC sampling) with a fixed 200-item prompt set.
- Replace the diary FTS5 recall with hybrid (FTS5 + embedding) retrieval, keeping the CLI.
- Port `smoke_test.py` to a GitHub Action (CPU, <5 min).

## Running the tests locally

```bash
python code/llm/model.py                     # parameter counts
python code/llm/smoke_test.py --steps 20     # CPU, ~1 min
python code/llm/smoke_test.py --resume-check # kill -9 resume determinism
```

## Reporting a bug

Please include: the stage, the command, the last 30 log lines, GPU model, and
`torch.__version__`. If it is a *number* that looks wrong, say what you expected and why.
