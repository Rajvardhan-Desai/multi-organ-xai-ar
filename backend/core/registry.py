# backend/core/registry.py
import os
from typing import Dict

def discover_model_artifacts(model_dir: str) -> Dict[str, str]:
    """
    Finds model.joblib, lut_parsed.csv (or lut.csv), and cn_reference.joblib in the given dir.
    Returns dict with keys: model, lut, (optional) cnref
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    candidates = os.listdir(model_dir)
    # model
    model = next((os.path.join(model_dir, f) for f in candidates if f.lower().endswith(".joblib") and "model" in f.lower()), None)
    if not model:
        # fallback: any *.joblib
        model = next((os.path.join(model_dir, f) for f in candidates if f.lower().endswith(".joblib")), None)
    if not model:
        raise FileNotFoundError("No joblib model found in model_dir")

    # LUT parsed preferred
    lut = next((os.path.join(model_dir, f) for f in candidates if f.lower() == "lut_parsed.csv"), None)
    if not lut:
        lut = next((os.path.join(model_dir, f) for f in candidates if f.lower() == "lut.csv"), None)
    if not lut:
        raise FileNotFoundError("No LUT CSV (lut_parsed.csv or lut.csv) found in model_dir")

    # CN reference (optional)
    cnref = next((os.path.join(model_dir, f) for f in candidates if f.lower().endswith(".joblib") and "cn" in f.lower()), None)

    out = {"model": model, "lut": lut}
    if cnref:
        out["cnref"] = cnref
    return out
