# backend/core/model_io.py
from typing import Any, Dict, Optional, Tuple
import joblib

# common keys people use when they save {"model": ..., ...}
_CANDIDATE_KEYS = (
    "model", "estimator", "clf", "pipeline", "sk_model", "sk_estimator"
)

def _has_predict(obj: Any) -> bool:
    return hasattr(obj, "predict")

def unwrap_estimator(obj: Any) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Accepts:
      - a bare sklearn estimator/pipeline
      - a tuple of (estimator, meta_dict)
      - a dict containing an estimator under common keys
    Returns:
      (estimator, meta_dict_or_None)
    """
    # bare estimator
    if _has_predict(obj):
        return obj, None

    # tuple pattern
    if isinstance(obj, tuple) and obj:
        est = obj[0]
        meta = obj[1] if len(obj) > 1 and isinstance(obj[1], dict) else None
        if _has_predict(est):
            return est, meta

    # dict pattern
    if isinstance(obj, dict):
        for k in _CANDIDATE_KEYS:
            if k in obj and _has_predict(obj[k]):
                return obj[k], obj

    raise TypeError(
        "Could not unwrap a scikit-learn estimator from the loaded object. "
        "Expected an object with .predict, or a dict/tuple containing one."
    )

def load_joblib_then_unwrap(path: str):
    """Load any joblib artifact and return (estimator, meta_dict_or_None)."""
    obj = joblib.load(path, mmap_mode="r")
    return unwrap_estimator(obj)
