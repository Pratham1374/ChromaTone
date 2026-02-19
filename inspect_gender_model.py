# backend/inspect_gender_model.py
import pickle, joblib, sys
from pathlib import Path

p = Path("models/gender_rf.pkl")
obj = None
try:
    obj = joblib.load(p)
    loader = "joblib"
except Exception as e:
    print(f"joblib load failed: {e}")
    with open(p, "rb") as f:
        obj = pickle.load(f)
    loader = "pickle"

print(f"\nLoaded with: {loader}")
print(f"Top-level type: {type(obj)}")

def describe(model, label="model"):
    import numpy as np
    print(f"\n--- {label} ---")
    print("type:", type(model))
    for name in ["n_features_in_", "classes_", "estimators_"]:
        v = getattr(model, name, None)
        if name == "estimators_" and v is not None:
            print("estimators_: len =", len(v), "| first type:", type(v[0]))
            nf = getattr(v[0], "n_features_in_", None)
            print(" first.estimator.n_features_in_ =", nf)
        else:
            if isinstance(v, (list, tuple)):
                print(name, "len =", len(v))
            else:
                print(name, "=", v)

if isinstance(obj, dict):
    for k, v in obj.items():
        print(f"dict key: {k} -> type {type(v)}")
    rf  = obj.get("rf") or obj.get("model")
    sc  = obj.get("scaler")
    le  = obj.get("label_encoder")
    if rf is not None: describe(rf, "rf")
    if sc is not None: describe(sc, "scaler")
    if le is not None: describe(le, "label_encoder")
else:
    describe(obj, "top_level_model")
