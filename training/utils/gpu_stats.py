"""Snapshot de GPU vía amd-smi para los logs de ejecución."""
from __future__ import annotations

import json
import shutil
import subprocess


def gpu_snapshot() -> dict:
    if shutil.which("amd-smi") is None:
        return {"available": False}
    try:
        out = subprocess.run(
            ["amd-smi", "metric", "--json"], capture_output=True, text=True, timeout=10
        )
        return {"available": True, "metrics": json.loads(out.stdout)}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)}
