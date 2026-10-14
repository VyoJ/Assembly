"""A model that holds a class defined in train.py (module __main__) must still load in another process,
which is what package.load_predictor does at deploy time."""

import subprocess
import sys
import types

import numpy as np

from ml.models import export


def test_model_with_main_class_loads_in_a_fresh_process(tmp_path):
    main = sys.modules["__main__"]
    cls = types.new_class("AgentModel", exec_body=lambda ns: ns.update(
        predict=lambda self, X: np.asarray(X).sum(axis=1) * self.k, __module__="__main__"))
    main.AgentModel = cls  # as if train.py defined it
    try:
        model = cls()
        model.k = 2.0
        export.save_model(model, str(tmp_path / "model"))
    finally:
        delattr(main, "AgentModel")

    code = (
        "from ml.models import export; import numpy as np; "
        f"m = export.load_saved({str(tmp_path / 'model')!r}); "
        "print(float(np.ravel(m.predict(np.ones((1, 3))))[0]))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == "6.0"
