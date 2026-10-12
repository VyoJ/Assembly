# Assembly / ForgeAgent

Autonomous agent that ingests industrial data, trains models on AMD ROCm,
exports them to ONNX and deploys them. See `docs/`.

- `training/`      : training and export (`training` branch)
- `orchestration/` : agent/LLM that calls the training tools (other teammate)
- `docs/datasets.md` : example datasets

## training <-> orchestration contract

```python
from training import train_model, capabilities, TrainConfig

capabilities()   # {"modalities": [...], "models": ["timeseries/classification/xgboost", ...]}
cfg = TrainConfig(modality="timeseries", task="classification", model="xgboost")
result = train_model("data/cwru", cfg)
result.to_dict()  # JSON: status, model_path, onnx_path, metrics, device, gpu_stats, log, error
```

`train_model` never raises towards the agent: it returns a `TrainResult` with
`status="error"` and a readable `error` if something fails.
