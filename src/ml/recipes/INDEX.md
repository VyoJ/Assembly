# Recipes

Working reference scripts the agent can read with `read_recipe(name)`. They are examples, not
templates: borrow what helps. Each must run end to end against `harness.py`.

TODO(ML): write these (one line each here: file, what it shows, when to use it).

## Sensors / tabular
- `xgb_window_features.py`: XGBoost on per-unit rolling-window features; fast baseline for scalar or binary targets.
- `lstm_sequence.py`: LSTM/GRU over fixed-length windows, budget-aware training loop with internal early stopping.
- `patchtst_hf.py`: PatchTST via `transformers` for longer histories.
- `threshold_calibration.py`: tune the alarm threshold for a false-alarm-rate KPI on an internal split.
- `optuna_inside_train.py`: time-boxed Optuna sweep inside train.py.

## Vision
- `yolo_finetune.py`: fine-tune Ultralytics YOLO on a data.yaml, export ONNX, image-level scoring.
- `sam3_zero_shot.py`: SAM 3 text prompts as a zero-shot detector.
- `sam3_pseudolabel_then_yolo.py`: SAM 3 labels the training images, then YOLO trains on them.
- `patchcore_knn.py`: anomaly scoring from frozen `timm` features + kNN (normal-only training data).

## Ideas worth trying
- Asymmetric loss that penalises late failure predictions more than early ones.
- Piecewise-linear RUL target (cap at ~125 cycles).
- `imgsz` 1024 + tiling for small defects; freeze backbone layers when data is tiny.
- SAM 3 prompt wording ("scratch" vs. "surface scratch on metal nut") and score threshold.
