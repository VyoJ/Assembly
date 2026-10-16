# Recipes

Reference scripts the agent can read with `read_recipe(name)`. They are examples, not templates:
borrow what helps. They use only the `harness.py` API described in `program.md`.

## Available
- `xgb_baseline.py`: XGBoost in an sklearn Pipeline for scalar / binary_score / class_probs. Budget-aware,
  early stopping on an internal group split, F1 threshold. **Start here for sensor tasks.**
- `threshold_calibration.py`: choose the alarm threshold from the KPIs (max recall under a false-alarm
  cap), and how to shift a scalar prediction for early-warning KPIs.

## Ideas worth trying (no recipe yet)
- History features (per-unit rolling mean / std / slope over the last N cycles) belong in `prepare.py`
  (design phase), never in `train.py`: the deployed model sees one row of `X` at a time. In `train.py`,
  check which ones `prepare.py` already built (read it with `read_workspace_file`).
- Piecewise-linear RUL target (cap at ~125 cycles) and an asymmetric loss that penalises late predictions.
- LightGBM as a faster alternative; small Optuna sweep sized with `harness.time_left()`.
- LSTM / 1D-CNN over fixed windows when trees plateau (torch; must output the final prediction).
- Vision: Ultralytics YOLO fine-tune from `harness.weights("yolo11n.pt")`; `imgsz` 1024 + tiling for
  small defects; freeze backbone layers when data is tiny.
