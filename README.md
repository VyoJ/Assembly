# Assembly / ForgeAgent

Agente autónomo que ingiere datos industriales, entrena modelos sobre AMD ROCm,
los exporta a ONNX y los despliega. Ver `docs/`.

- `training/`      : entrenamiento y export (rama `training`)
- `orchestration/` : agente/LLM que llama a las tools de training (otra persona)
- `docs/datasets.md` : datasets de ejemplo

## Contrato training <-> orchestration

```python
from training import train_model, capabilities, TrainConfig

capabilities()   # {"modalities": [...], "models": ["timeseries/classification/xgboost", ...]}
cfg = TrainConfig(modality="timeseries", task="classification", model="xgboost")
result = train_model("data/cwru", cfg)
result.to_dict()  # JSON: status, model_path, onnx_path, metrics, device, gpu_stats, log, error
```

`train_model` nunca lanza hacia el agente: devuelve `TrainResult` con
`status="error"` y `error` legible si algo falla.
