# ML Plan (training half)

Owners: **Aryan**, **Samu**, **Andres**. Orchestration (agent, API, frontend): **Vyoman, Yashwanth, Arvind**.
Base branch: `init`. Background: [README.md](README.md), [PLAN.md](PLAN.md), [src/core/contracts.py](src/core/contracts.py).

## What we agreed

- **General, not fixed pipelines.** Like [karpathy/autoresearch](https://github.com/karpathy/autoresearch): the agent writes `prepare.py` (builds the dataset) and keeps rewriting `train.py`. We build the fixed lab around it: data access, validation, scoring, running, packaging. The agent can't change any of it.
- **One interface.** The agent only calls [src/ml/api.py](src/ml/api.py), using the Pydantic types in `contracts.py`. Change `contracts.py` only after checking with Vyoman.
- **`training` branch:** `gpu_stats.py` and `docs/datasets.md` are copied into `init` (`onnx_export.py` was an empty stub; export lives in `ml/models/export.py`). The branch can be deleted.
- **Order:** the sensor path end to end first (C-MAPSS → XGBoost → package). Then vision (MVTec → YOLO). SAM 3, PatchTST, audio and video only if there's time.

## Split

| Owner | Files (`src/ml/`) | Done when |
|---|---|---|
| **Samu** (data in) | `scripts/download_data.py` + `catalog.py` (`read_stream`, `read_labels`, `peek_stream`) | C-MAPSS FD001 streams + `maintenance_log` load as DataFrames; MVTec `metal_nut` second |
| | `harness.py` **prepare side**: `catalog`, `read_stream`, `read_labels`, `write_dataset` | A hand-written `prepare.py` for `specs/rul_example.json` writes `data/` + hidden labels in the layout below |
| **Andres** (data out + checks) | `harness.py` **train side**: `load_train`, `load_val_inputs`, `load_test_inputs`, `save_model`, `finish` | A hand-written `train.py` (XGBoost) writes `exp_NNN/` predictions, timings and model in the layout below |
| | `validators.py` → `validate(ws)` | The 7 `CHECKS` produce clear messages the LLM can act on, plus a data card |
| **Aryan** | `runner.py` (all 4 functions) | Subprocess + env vars + timeout + scoring + `record_experiment`; crash, timeout and invalid output handled |
| | `scoring.py` → `compute_metrics` | Every metric in `METRIC_REGISTRY` for `scalar` + `binary_score` first, then `boxes` |
| | `package.py` + `models/export.py` | Best model → `package/` (ONNX + `manifest.json`); `load_predictor().predict()` works |
| | `program.md` + `recipes/` | The real LLM gets from `exp_000` to KPIs met on the RUL spec |
| Later | `models/features.py`, `splits.py` (Samu); `models/calibrate.py`, `profiling.py` (Andres) | Whoever is free first |

## The seam: workspace file layout (agree this first)

`harness.py` (Samu: `data/`, Andres: `exp_NNN/`) writes these files; `runner.py`/`scoring.py` (Aryan) read them. Every row has a `sample_id`, so predictions are matched to labels by id, never by row order.

```
runs/{run}/{task}/data/                       written by harness.write_dataset (prepare.py)
    train_inputs.parquet   sample_id + feature columns   (images: sample_id, path)
    train_labels.parquet   sample_id + label columns (format below)
    val_inputs.parquet     test_inputs.parquet
    meta.json              {"prediction_format", "classes"?, "groups": {split: [ids]}, "time_col"?,
                            "early_warning_horizon"?, ...anything validators need}
runs/{run}/_labels/{task}/                    hidden; only prepare.py sees ASSEMBLY_LABELS_DIR
    val_labels.parquet     test_labels.parquet

runs/{run}/{task}/exp_NNN/                    written by train.py via harness
    model/                 written by save_model: model.onnx | model.joblib + model.json (preprocess, threshold)
    val_predictions.*      written by finish()
    test_predictions.*
    timings.json           {"p50_latency_ms": .., "p95_latency_ms": ..}
    predict_meta.json      {"threshold": ..}   (binary_score)
```

Label and prediction columns per `prediction_format`:

| Format | labels | predictions |
|---|---|---|
| `scalar` | `y` | `pred` |
| `binary_score` | `y` (0/1) | `score` (+ threshold in `predict_meta.json`) |
| `class_probs` | `y` (class index) | `p_0 … p_{k-1}` |
| `boxes` | `*.json`: `{sample_id: [[x1,y1,x2,y2,class], …]}` | `*.json`: `{sample_id: [[x1,y1,x2,y2,score,class], …]}` |
| `masks` | `*.json`: `{sample_id: mask_png_path}` | `*.json`: `{sample_id: [{"mask": png_path, "score": s}, …]}` |

**Early-warning horizon:** `prepare.py` passes `meta={"early_warning_horizon": 30}` to `write_dataset`; scoring
and the deployed predictor read it from `data/meta.json` (default 30). `exp_NNN/timings.json` may also carry
`"peak_mem_gb"` (`torch.cuda.max_memory_allocated() / 1e9`); the runner reports it.

## Harness Python API (what `train.py`, `program.md` and the recipes assume)

Samu + Andres implement these in `src/ml/harness/harness.py`; Aryan's code and `program.md` rely on them.

| Function | Returns / does |
|---|---|
| `load_train()` | `(X, y)`: `X` DataFrame of features **indexed by `sample_id`** (no `sample_id` column), `y` Series, same index |
| `load_val_inputs()` / `load_test_inputs()` | `X` only, same shape conventions |
| `groups(split="train")` | **new**: Series of group ids (e.g. engine unit) indexed by `sample_id`, from `meta.json["groups"]` |
| `save_model(model, **kw)` | `ml.models.export.save_model(model, EXP_DIR / "model", example_input=load_train()[0].head(8), **kw)` |
| `finish(predict_fn, threshold=None)` | Calls `predict_fn(X)` on val and test, checks the format, writes `{val,test}_predictions.*` with `sample_id` from `X.index`, `timings.json` (per-sample p50/p95 ms), `predict_meta.json` (`{"threshold": t}`) |

`ml.models.selfcheck.selfcheck(fmt, y_true, y_pred, metrics)` gives `train.py` the exact scorer definitions for internal checks.

## Status (Aryan's part, 14 Oct)

Done and tested (`tests/test_runner.py`, against stand-in scripts that write the layout above directly):
- `scoring.compute_metrics`: all formats (scalar, binary, class_probs, boxes, masks) + latency, matched by `sample_id`;
  bad predictions → `invalid_output` with a message the agent can act on.
- `runner.py`: `run_prepare`, `run_experiment` (subprocess, secrets stripped, hard kill at 2x budget + 60 s,
  crash / timeout / invalid_output, live `key=value` progress), `full_train`, `final_test`.
- `models/export.py` + `package.py`: joblib (sklearn / XGBoost / LightGBM), torch → ONNX, YOLO → ONNX; ONNX Runtime
  prefers MIGraphX → ROCm → CPU. `Predictor.predict(sample)` returns the same keys as the mock; `{"replay": i}` replays
  held-out test rows for the live demo.
- `program.md` (deployable-model rule, prediction formats, progress lines), recipes `xgb_baseline.py`,
  `threshold_calibration.py`; `models/selfcheck.py`; `models/gpu_stats.py` + `docs/datasets.md` copied from `training`.

Blocked on: `harness.py` and `validators.validate` (the runner calls both), and real C-MAPSS data.

## Timeline (submission 18 Oct; check the exact lablab deadline)

| Day | Samu + Andres | Aryan |
|---|---|---|
| **Tue 13** | Download C-MAPSS; start `catalog.py` | Lock this layout; copy files from `training`, delete it |
| **Wed 14** | `harness.py` for tabular; `validators.py` core checks | `scoring.py` (scalar, binary); `runner.run_prepare` + `run_experiment` |
| **Thu 15** | `harness.save_model` (calls `models/export.py`); remaining validator checks | `full_train`, `final_test`, `program.md`, `package.py`; **first real run with the LLM** on the RUL spec |
| **Fri 16** | MVTec in catalog/harness | `boxes` metrics; YOLO recipe; ONNX Runtime with the MIGraphX/ROCm backend on the AMD box; tune `program.md` |
| **Sat 17** | Bug fixes; record the demo run (`BUDGET_PROFILE=demo`) | Same |
| **Sun 18** | Video, slides, README, submission | Same |

Milestone that matters most: **by Thu 15, `ML_MOCK=0 uv run python scripts/run_research.py --spec specs/rul_example.json --n 10` runs for real.**

## Compute

- **On laptops:** `ML_MOCK=1 LLM_MOCK=1` for plumbing. C-MAPSS + XGBoost is small enough to run for real on a laptop CPU.
- **LLM:** AMD Token Factory endpoint in `LLM_BASE_URL` (no credits needed) until someone has a GPU box.
- **GPU box:** whoever gets AMD Cloud credits first sets up one shared box (ROCm, `uv sync --group ml`). It's for vision, final runs and the demo. Destroy it when idle.

## Starter prompt for Claude Code

> Read README.md, PLAN.md (section on the ML lab), ML_PLAN.md, src/core/contracts.py and everything in src/ml/.
> Implement `<your files>` following the workspace file layout in ML_PLAN.md exactly. Don't change contracts.py or
> api.py. Add unit tests in tests/. Goal: `ML_MOCK=0 uv run python scripts/run_research.py --spec specs/rul_example.json`
> gets further than before.

## Questions for the orchestration team

1. Has anyone received AMD Cloud credits?
2. LLM: self-hosted vLLM or Token Factory? (The Evolus track requires our own vLLM on AMD Developer Cloud.)
3. Which track(s) are we submitting to: Intelligent Industry only, or Evolus too?
