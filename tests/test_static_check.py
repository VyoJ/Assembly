"""Through runner.static_check (a subprocess, as the pipeline runs it): importing xgboost into the test process
would make later torch tests segfault on macOS (two libomp; see ml.package process isolation)."""

import importlib.util

import pytest

from ml import runner

pytestmark = pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="xgboost not installed")


def test_flags_kwargs_the_installed_version_rejects():
    code = (
        "import xgboost as xgb\n"
        "model = xgb.XGBRegressor(n_estimators=10)\n"
        "model.fit(X, y, early_stopping_rounds=5)\n"
        "cb = xgb.callback.EarlyStopping(rounds=5, restore_best_weight=True)\n"
    )
    problems = runner.static_check(code)
    assert any("XGBRegressor.fit() has no argument 'early_stopping_rounds'" in p for p in problems)
    assert any("EarlyStopping() has no argument 'restore_best_weight'" in p for p in problems)


def test_valid_code_and_kwargs_callables_pass():
    code = (
        "import xgboost as xgb\n"
        "from sklearn.pipeline import Pipeline\n"
        "X = y = unknown_obj = None\n"
        "model = xgb.XGBRegressor(n_estimators=10, early_stopping_rounds=5, anything_goes=1)\n"  # __init__ has **kwargs
        "model.fit(X, y, eval_set=[(X, y)], verbose=False)\n"
        "pipe = Pipeline([('m', model)])\n"
        "unknown_obj.fit(bogus=1)\n"  # unresolvable: not checked
    )
    assert runner.static_check(code) == []


def test_runner_static_check_end_to_end():
    assert runner.static_check("import xgboost as xgb\nxgb.callback.EarlyStopping(rounds=3, nope=1)\n")
    assert runner.static_check("import xgboost as xgb\nxgb.callback.EarlyStopping(rounds=3)\n") == []
