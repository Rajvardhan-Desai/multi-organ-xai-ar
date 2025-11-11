# backend/core/heart_predict.py
from __future__ import annotations
import os, json
from typing import Any, Dict, List
import numpy as np
import joblib

from .heart_features import extract_features_from_pair

def _load_json(path:str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _align_features(feat:Dict[str,float], x_cols:List[str]) -> np.ndarray:
    x = np.zeros((len(x_cols),), dtype=float)
    for i, c in enumerate(x_cols):
        x[i] = float(feat.get(c, 0.0))
    return x

def predict_heart_cardio(model_dir: str, ed_mask_bytes: bytes, es_mask_bytes: bytes, want_xai: bool) -> Dict[str, Any]:
    """
    Inference entry for organ=heart, disease=cardiomyopathy.
    Expects ED/ES *mask* NIfTIs (labels: 0 bg, 1 LV, 2 RV, 3 LV-myo).
    Returns classifier prediction + per-segment (AHA16) scores for GLB coloring.
    """
    # Load artifacts
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    x_cols = _load_json(os.path.join(model_dir, "x_cols.json"))
    label_map = _load_json(os.path.join(model_dir, "xgb_label_map.json"))["classes"]

    # Features
    feat, seg_scores = extract_features_from_pair(ed_mask_bytes, es_mask_bytes)
    X = _align_features(feat, x_cols).reshape(1, -1)
    Xs = scaler.transform(X)

    # Predict
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xs)[0]
    else:
        y = int(clf.predict(Xs)[0])
        proba = np.eye(len(label_map))[y]
    pred_idx = int(np.argmax(proba))
    pred_label = label_map[pred_idx]
    proba_map = {label_map[i]: float(proba[i]) for i in range(len(label_map))}

    # Simple XAI fallback (gain) → map segment-related features back to AHA ids
    xai = None
    if want_xai:
        try:
            booster = clf.get_booster()
            gain = booster.get_score(importance_type="gain")
            importances = [float(gain.get(f"f{i}", 0.0)) for i in range(len(x_cols))]
            order = np.argsort(importances)[::-1]
            top_regions = []
            for i in order[:16]:
                nm = x_cols[i]
                if nm.startswith("SEG") and ("thkED" in nm or "dThk" in nm or "thkES" in nm):
                    sid = int(nm.split("_")[0][3:])
                    top_regions.append({"label_id": sid, "label_name": f"AHA{sid}", "score": float(importances[i])})
            xai = {"method": "gain", "top_regions": top_regions}
        except Exception:
            xai = {"method": "none"}

    return {
        "prediction": pred_label,
        "proba": proba_map,
        "used_features": x_cols,
        "segment_scores": {str(k): float(v) for k, v in seg_scores.items()},
        "xai": xai,
    }
