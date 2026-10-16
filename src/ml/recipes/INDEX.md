# Recipes

Reference scripts the agent can read with `read_recipe(name)`. They are examples, not templates:
borrow what helps. They use only the `harness.py` API described in `program.md`.

## Available
- `xgb_baseline.py`: XGBoost in an sklearn Pipeline for scalar / binary_score / class_probs. Budget-aware,
  early stopping on an internal group split, F1 threshold. **Start here for sensor tasks.**
- `threshold_calibration.py`: choose the alarm threshold from the KPIs (max recall under a false-alarm
  cap), and how to shift a scalar prediction for early-warning KPIs.

## API notes for the installed versions (xgboost 3.x, lightgbm 4.x, scikit-learn 1.9)
- XGBoost: `early_stopping_rounds`, `eval_metric` and `callbacks` go in the CONSTRUCTOR
  (`xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=50)`), not in `fit()`; `fit(X, y, eval_set=[(Xv, yv)],
  verbose=False)`. A model built with `early_stopping_rounds` needs an `eval_set` every time it is fit: to refit on
  all data, build a new model with `n_estimators=best_iteration + 1` and no early stopping.
  `xgb.callback.EarlyStopping(rounds=..., save_best=...)`.
- LightGBM: early stopping is a callback in `fit()`: `fit(X, y, eval_set=[(Xv, yv)],
  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])`; there is no `early_stopping_rounds` argument.
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
- LSTM / 1D-CNN over fixed windows when trees plateau (torch; must output the final prediction).
- Vision: Ultralytics YOLO fine-tune from `harness.weights("yolo11n.pt")`; `imgsz` 1024 + tiling for
  small defects; freeze backbone layers when data is tiny.
