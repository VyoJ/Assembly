"""Recipe: XGBoost baseline for tabular / sensor-feature tasks (scalar, binary_score, class_probs).

Budget-aware (stops adding trees when time runs low), early stopping on an internal GROUP split,
model = sklearn Pipeline so preprocessing ships with it.
"""

import time

import harness
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

task = harness.task()
fmt = task["prediction_format"]
X, y = harness.load_train()
groups = harness.groups("train")  # e.g. engine unit: never split one unit across train / internal-val

tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, groups))
X_tr, X_va, y_tr, y_va = X.iloc[tr_idx], X.iloc[va_idx], y.iloc[tr_idx], y.iloc[va_idx]

params = {
    "n_estimators": 2000, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
    "tree_method": "hist", "early_stopping_rounds": 50, "random_state": 0,
}
deadline = time.monotonic() + 0.8 * harness.time_left()


class StopAtBudget(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        if epoch % 25 == 0:
            name, hist = next(iter(evals_log["validation_0"].items()))
            print(f"epoch={epoch} {name}={hist[-1]:.4f} time_left={harness.time_left():.0f}")
        return time.monotonic() > deadline  # True = stop


if fmt == "scalar":
    model = xgb.XGBRegressor(objective="reg:squarederror", callbacks=[StopAtBudget()], **params)
elif fmt == "binary_score":
    pos = max(float(y_tr.mean()), 1e-3)
    model = xgb.XGBClassifier(eval_metric="aucpr", scale_pos_weight=(1 - pos) / pos, callbacks=[StopAtBudget()], **params)
else:  # class_probs
    model = xgb.XGBClassifier(eval_metric="mlogloss", callbacks=[StopAtBudget()], **params)

imputer = SimpleImputer(strategy="median").fit(X_tr)
model.fit(imputer.transform(X_tr), y_tr, eval_set=[(imputer.transform(X_va), y_va)], verbose=False)
model.set_params(callbacks=None)  # the budget callback uses harness; don't ship it inside the model

# Refit-free: wrap the fitted pieces as one deployable pipeline (takes raw feature columns).
pipe = Pipeline([("impute", imputer), ("model", model)])

threshold = None
if fmt == "binary_score":
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(y_va, pipe.predict_proba(X_va)[:, 1])
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    threshold = float(t[np.argmax(f1[:-1])])  # see threshold_calibration.py for KPI-driven thresholds
    print(f"threshold={threshold:.3f}")


def predict_fn(X):
    if fmt == "scalar":
        return pipe.predict(X)
    if fmt == "binary_score":
        return pipe.predict_proba(X)[:, 1]
    return pipe.predict_proba(X)


harness.save_model(pipe)
harness.finish(predict_fn, threshold=threshold)
