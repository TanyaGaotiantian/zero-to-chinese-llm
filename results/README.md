# Raw results

Everything here is copied straight from the run directory — no retouching.

| File | What |
|---|---|
| `stats.json` | corpus statistics (`data/processed/stats.json`) |
| `funnel.json` | cleaning funnel (`data/interim/funnel.json`) |
| `125m/train.csv` | 125M training log: step, loss, lr, grad-norm, tok/s, batch sha1 |
| `125m/samples.txt` | base-model generations |
| `360m/train.csv` | 360M training log |
| `360m/samples.txt` | 360M generations |
| `sft/log.txt` | SFT log (fixed shift) |
| `sft_buggy.log` | the failing run where loss collapsed to 0.0005 (kept on purpose) |
| `verify_resume.txt` | `verify_resume.py` output for the kill -9 test |
| `smoke.log` | 20-step smoke test |
| `prep.log` | full corpus pipeline log |

Note: `wall_s` / `tok_per_s` in the CSVs are per-log-interval and include a warmup step, which is
why the first row is slower than the steady state.
