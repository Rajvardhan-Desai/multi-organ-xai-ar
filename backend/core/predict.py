# core/predict.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.base import ClassifierMixin

from .features import extract_roi_features

def _feature_name_to_label_id(name: str) -> int | None:
    if name.startswith("vol_"):
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return None
    return None

def _align_importances(importances: np.ndarray, x_cols: List[str]) -> Optional[np.ndarray]:
    """Return importances if it matches x_cols length; else None."""
    try:
        imp = np.asarray(importances).ravel()
        return imp if imp.shape[0] == len(x_cols) else None
    except Exception:
        return None

def _top_regions_from_importance(
    importances: np.ndarray,
    x_cols: List[str],
    lut: Dict[int, str],
    k: int = 10,
) -> List[Dict[str, Any]]:
    pairs: List[Tuple[str, float, int | None]] = []
    for imp, col in zip(importances, x_cols):
        pairs.append((col, float(imp), _feature_name_to_label_id(col)))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    out: List[Dict[str, Any]] = []
    for col, score, lid in pairs[:k]:
        out.append({
            "label_id": lid,
            "label_name": lut.get(lid, col) if lid is not None else col,
            "score": float(score),
        })
    return out

def _fallback_top_regions_by_volume(
    lut: Dict[int, str],
    col_to_val: Dict[str, float],
    icv: float,
    k: int = 10,
) -> List[Dict[str, Any]]:
    # Rank by normalized volume
    items: List[Tuple[int, float]] = []
    for col, v in col_to_val.items():
        lid = _feature_name_to_label_id(col)
        if lid is None:
            continue
        score = float(v) / (float(icv) + 1e-9)
        items.append((lid, score))
    items.sort(key=lambda t: t[1], reverse=True)
    top = []
    for lid, score in items[:k]:
        top.append({
            "label_id": lid,
            "label_name": lut.get(lid, f"vol_{lid}"),
            "score": float(score),
        })
    return top

def predict(
    bundle: Dict[str, Any],
    seg_img: nib.Nifti1Image,
    lut: Dict[int, str],
    produce_xai: bool = False,
    class_name_map: Optional[Dict[str, str]] = None,  # NEW
) -> Dict[str, Any]:
    """
    bundle: {"model": estimator, "x_cols": [...], "classes": [...]}
    seg_img: MALPEM segmentation (labels 1..138)
    lut: {label_id: label_name}
    class_name_map: maps raw estimator labels to friendly e.g. {"0":"CN","1":"AD"}
    """
    model: ClassifierMixin = bundle["model"]
    x_cols: List[str] = [str(c) for c in bundle["x_cols"]]
    classes_bundle: List[str] = [str(c) for c in bundle["classes"]]

    # Features
    feats_arr, icv = extract_roi_features(seg_img, lut, return_type="array")
    label_ids = sorted(lut.keys())
    col_to_val: Dict[str, float] = {f"vol_{lid}": float(v) for lid, v in zip(label_ids, feats_arr.tolist())}
    x_row = [float(col_to_val.get(col, 0.0)) for col in x_cols]
    X = pd.DataFrame([x_row], columns=x_cols)

    # Predict
    raw_pred_label: str
    proba: Dict[str, float]

    if hasattr(model, "predict_proba"):
        proba_vec = model.predict_proba(X)[0]
        est_classes = getattr(model, "classes_", np.array(classes_bundle, dtype=object))
        # map to strings
        est_classes = np.array([str(c) for c in est_classes], dtype=object)
        raw_pred_label = str(est_classes[int(np.argmax(proba_vec))])
        proba = {str(c): float(p) for c, p in zip(est_classes, proba_vec)}
    else:
        yhat = model.predict(X)[0]
        raw_pred_label = str(yhat)
        # fabricate proba 1/0
        proba = {c: (1.0 if str(c) == raw_pred_label else 0.0) for c in classes_bundle}

    # Friendly label mapping
    def map_name(s: str) -> str:
        if class_name_map and s in class_name_map:
            return class_name_map[s]
        # fallbacks
        if s in {"0", "1"} and set(classes_bundle) & {"CN", "AD"}:
            return "CN" if s == "0" else "AD"
        return s

    pred_label = map_name(raw_pred_label)
    proba = {map_name(k): v for k, v in proba.items()}

    # XAI
    top_regions: List[Dict[str, Any]] = []
    xai_payload = None
    if produce_xai:
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = _align_importances(np.asarray(model.feature_importances_), x_cols)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            if coef.ndim == 2:
                importances = _align_importances(np.mean(np.abs(coef), axis=0), x_cols)
            else:
                importances = _align_importances(np.abs(coef).ravel(), x_cols)

        if importances is not None:
            top_regions = _top_regions_from_importance(importances, x_cols, lut, k=10)
            xai_payload = {"method": "feature_importance", "top_regions": top_regions}
        else:
            # robust fallback: rank by normalized volume
            top_regions = _fallback_top_regions_by_volume(lut, col_to_val, icv, k=10)
            xai_payload = {"method": "normalized_volume_fallback", "top_regions": top_regions}
    else:
        # still give useful top regions (volume-based) if caller didn’t request XAI
        top_regions = _fallback_top_regions_by_volume(lut, col_to_val, icv, k=10)
        xai_payload = None

    return {
        "prediction": pred_label,
        "proba": proba,
        "icv_mm3": float(icv),
        "used_features": x_cols,
        "top_regions": top_regions,
        "xai": xai_payload,
    }
