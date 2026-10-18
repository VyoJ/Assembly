# Recipes

Reference scripts the agent can read with `read_recipe(name)`. They are examples, not templates:
borrow what helps. They use only the `harness.py` API described in `program.md`.

## Dataset patterns for prepare.py (design phase)
- `prepare_sensor_history.py`: table streams joined per unit, target from failure events (cycles to failure,
  capped), CAUSAL per-unit rolling / delta history features built here, split by unit.
- `prepare_images_binary.py`: camera images + per-image QC labels -> defective or not; one image per part, so a
  random split stratified by defect type (`meta={"split": "random"}`); don't use folder names as labels.
- `prepare_images_boxes.py`: same, with boxes from the QC pixel masks (one box per connected defect region).

## Available (train.py)
- `xgb_baseline.py`: XGBoost in an sklearn Pipeline for scalar / binary_score / class_probs. Budget-aware,
  early stopping on an internal group split, F1 threshold. **Start here for sensor tasks.**
- `image_anomaly.py`: camera / visual inspection on CPU with a pretrained ResNet-18. `binary_score`:
  PatchCore-style anomaly score from a memory bank of normal patches (no defect examples needed);
  `class_probs`: embeddings + logistic regression. **Start here for image tasks.**
- `yolo_boxes.py`: `boxes` tasks (where is the defect) by fine-tuning YOLO11-nano from
  `harness.weights("yolo11n.pt")`; wall-clock training budget, offline, alarm threshold calibrated on an
  internal split. **Start here for box tasks.**
- `threshold_calibration.py`: choose the alarm threshold from the KPIs (max recall under a false-alarm
  cap), and how to shift a scalar prediction for early-warning KPIs.

## API notes for the installed versions (xgboost 3.x, lightgbm 4.x, scikit-learn 1.9)
- XGBoost: `early_stopping_rounds`, `eval_metric` and `callbacks` go in the CONSTRUCTOR
  (`xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=50)`), not in `fit()`; `fit(X, y, eval_set=[(Xv, yv)],
  verbose=False)`. A model built with `early_stopping_rounds` needs an `eval_set` every time it is fit: to refit on
  all data, build a new model with `n_estimators=best_iteration + 1` and no early stopping.
  `xgb.callback.EarlyStopping(rounds=..., save_best=...)`.
- LightGBM: early stopping is a callback in `fit()`: `fit(X, y, eval_set=[(Xv, yv)],
  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])`; there is no `early_stopping_rounds` and no `verbose`
  argument in `fit()`. Silence training logs with `verbose=-1` in the CONSTRUCTOR and `lgb.log_evaluation(0)`.
- scikit-learn: anything inside a `Pipeline` must be an estimator (subclass `BaseEstimator` + `RegressorMixin` /
  `TransformerMixin`, implement `fit` / `predict` or `transform`). To clip or transform the target, prefer
  `TransformedTargetRegressor` or a `FunctionTransformer` step; `FunctionTransformer(..., feature_names_out=...)`
  must return exactly those column names.

## Ideas worth trying (no recipe yet)
- History features (per-unit rolling mean / std / slope over the last N cycles) belong in `prepare.py`
  (design phase), never in `train.py`: the deployed model sees one row of `X` at a time. In `train.py`,
  check which ones `prepare.py` already built (read it with `read_workspace_file`).
- Piecewise-linear RUL target (cap at ~125 cycles) and an asymmetric loss that penalises late predictions.
- LightGBM as a faster alternative; small Optuna sweep sized with `harness.time_left()`.
- Raw vibration windows (array inputs): log band energies of the FFT, RMS / kurtosis / crest factor, and the
  ENVELOPE spectrum (abs of `scipy.signal.hilbert` of a band-passed window, then FFT) for bearing faults: ball
  faults are weak in the plain spectrum. Normalise per window so other motor loads still match.
- LSTM / 1D-CNN over fixed windows when trees plateau (torch; must output the final prediction).
- Vision: larger `image_size` (320+) or tiling for small defects; ensemble PatchKNN with the embedding
  classifier; for `boxes`, a larger `imgsz` or `yolo11s.pt` (if listed in `data_card.md`) with more time.
