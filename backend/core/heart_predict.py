# backend/core/heart_predict.py
from __future__ import annotations
import os, json
from typing import Any, Dict, List
import numpy as np
import joblib
import xgboost as xgb

from .heart_features import extract_features_from_pair

def _load_json(path:str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _align_features(feat:Dict[str,float], x_cols:List[str]) -> np.ndarray:
    x = np.zeros((len(x_cols),), dtype=float)
    for i, c in enumerate(x_cols):
        x[i] = float(feat.get(c, 0.0))
    return x

def _compat_patch_xgb(clf: Any) -> Any:
    """
    Make older, pickled XGBClassifier objects usable on newer xgboost:
    - add missing attributes commonly referenced by older wrappers
    - keep safe defaults (CPU, no label encoder)
    """
    # removed in >=1.7; some pickles still look it up
    if hasattr(clf, "__dict__") and "use_label_encoder" not in clf.__dict__:
        try: setattr(clf, "use_label_encoder", False)
        except Exception: pass

    # some versions expect these to exist
    defaults = {
        "gpu_id": -1,          # force CPU
        "tree_method": "auto",
        "n_jobs": 1,
        "verbosity": 1,
        "missing": np.nan,
    }
    for k, v in defaults.items():
        if not hasattr(clf, k):
            try: setattr(clf, k, v)
            except Exception: pass
    return clf

def _booster_from_model(clf: Any):
    # best: official API
    getb = getattr(clf, "get_booster", None)
    if callable(getb):
        try:
            return getb()
        except Exception:
            pass
    # very old pickles store a private _Booster
    bst = getattr(clf, "_Booster", None)
    return bst

def _proba_from_booster(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """
    Get probabilities from raw Booster, robust across versions.
    """
    if booster is None:
        raise RuntimeError("No Booster available in XGB model.")
    try:
        # fastest (newer xgboost)
        pred = booster.inplace_predict(X, validate_features=False)
    except Exception:
        # fallback (older)
        dm = xgb.DMatrix(X, missing=np.nan)
        pred = booster.predict(dm, output_margin=False)

    pred = np.asarray(pred)
    if pred.ndim == 1:
        # binary classifier typically returns p(positive)
        pred = np.c_[1.0 - pred, pred]
    return pred

def predict_heart_cardio(model_dir: str, ed_mask_bytes: bytes, es_mask_bytes: bytes, want_xai: bool) -> Dict[str, Any]:
    # Load artifacts
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    clf = _compat_patch_xgb(clf)

    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    x_cols = _load_json(os.path.join(model_dir, "x_cols.json"))
    label_map = _load_json(os.path.join(model_dir, "xgb_label_map.json"))["classes"]

    # Features (ED/ES masks → volumes/EF/segment thicknesses + AHA16 scores)
    feat, seg_scores = extract_features_from_pair(ed_mask_bytes, es_mask_bytes)
    X = _align_features(feat, x_cols).reshape(1, -1)
    Xs = scaler.transform(X)

    # Predict probabilities robustly
    try:
        proba = clf.predict_proba(Xs)[0]
    except Exception:
        bst = _booster_from_model(clf)
        proba = _proba_from_booster(bst, Xs)[0]

    # Normalize to label_map length (defensive)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 1:
        proba = proba.ravel()
    K = len(label_map)
    if proba.size != K:
        # pad / trim to match declared classes
        if proba.size < K:
            proba = np.pad(proba, (0, K - proba.size))
        else:
            proba = proba[:K]
    # renormalize
    s = proba.sum()
    if s > 0:
        proba = proba / s

    pred_idx = int(np.argmax(proba))
    pred_label = label_map[pred_idx]
    proba_map = {label_map[i]: float(proba[i]) for i in range(K)}

    # Optional XAI via Booster gain (best-effort, never crash)
    xai = None
    if want_xai:
        try:
            bst = _booster_from_model(clf)
            if bst is not None:
                gain = bst.get_score(importance_type="gain")
                importances = [float(gain.get(f"f{i}", 0.0)) for i in range(len(x_cols))]
                order = np.argsort(importances)[::-1]
                top_regions = []
                for i in order[:16]:
                    nm = x_cols[i]
                    if nm.startswith("SEG") and ("thkED" in nm or "thkES" in nm or "dThk" in nm):
                        sid = int(nm.split("_")[0][3:])
                        top_regions.append({
                            "label_id": sid,
                            "label_name": f"AHA{sid}",
                            "score": float(importances[i]),
                        })
                xai = {"method": "gain", "top_regions": top_regions}
            else:
                xai = {"method": "none"}
        except Exception:
            xai = {"method": "none"}

    return {
        "prediction": pred_label,
        "proba": proba_map,
        "used_features": x_cols,
        "segment_scores": {str(k): float(v) for k, v in seg_scores.items()},
        "xai": xai,
    }
