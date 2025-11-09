# core/models.py
from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import joblib

def load_model_bundle(path: str) -> Dict[str, Any]:
    """
    Load a saved model bundle (joblib).
    Accepts either:
      - a dict like {"model": estimator, "x_cols": [...], "classes": [...], ...}
      - a raw estimator, in which case we synthesize x_cols/classes defaults
        (but your saved bundle already has them).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"model bundle not found: {p}")

    obj = joblib.load(p)
    if isinstance(obj, dict) and "model" in obj:
        # Normalize x_cols to strings, classes to list of str
        if "x_cols" in obj:
            obj["x_cols"] = [str(x) for x in obj["x_cols"]]
        if "classes" in obj:
            obj["classes"] = [str(c) for c in obj["classes"]]
        return obj

    # Raw estimator: create a minimal bundle
    est = obj
    return {
        "model": est,
        "x_cols": [f"vol_{i}" for i in range(1, 139)],  # default 138 features
        "classes": ["CN", "AD"],  # default 2-class
    }
