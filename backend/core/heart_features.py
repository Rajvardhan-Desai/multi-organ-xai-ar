# backend/core/heart_features.py
from __future__ import annotations
import tempfile, os
from typing import Dict, Tuple
import numpy as np
import nibabel as nib

def _load_mask_from_bytes(b: bytes) -> np.ndarray:
    """Robust load for Windows: write to temp, then nib.load."""
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(b)
        path = tmp.name
    try:
        img = nib.load(path)
        arr = img.get_fdata().astype(np.int16)
    finally:
        try: os.remove(path)
        except Exception: pass
    return arr

def _volumes_mm3(mask: np.ndarray, spacing=(1.5,1.5,2.0)) -> tuple[float,float,float]:
    vx = float(spacing[0]*spacing[1]*spacing[2])
    v_lv = float((mask==1).sum()) * vx
    v_rv = float((mask==2).sum()) * vx
    v_myo= float((mask==3).sum()) * vx
    return v_lv, v_rv, v_myo

def _wall_thickness(mask: np.ndarray) -> np.ndarray:
    """Proxy thickness via EDT inside myocardium (label 3)."""
    try:
        from scipy.ndimage import distance_transform_edt
        myo = (mask==3).astype(np.uint8)
        return distance_transform_edt(myo)
    except Exception:
        # Fallback: zero thickness (keeps pipeline running)
        return np.zeros_like(mask, dtype=float)

def _aha16_bins(d:int,h:int,w:int) -> dict[int, np.ndarray]:
    z = np.linspace(0,1,d).reshape(d,1,1)
    z = np.broadcast_to(z,(d,h,w))
    yy = np.linspace(-1,1,h).reshape(1,h,1)
    yy = np.broadcast_to(yy,(d,h,w))
    xx = np.linspace(-1,1,w).reshape(1,1,w)
    xx = np.broadcast_to(xx,(d,h,w))
    theta = (np.arctan2(yy, xx) + np.pi) / (2*np.pi)  # [0,1)
    segs: dict[int,np.ndarray] = {}
    def ring(lo,hi): return (z>=lo)&(z<hi)
    def sectors(n):
        edges = np.linspace(0,1,n+1)
        return [ (theta>=edges[i]) & (theta<edges[i+1]) for i in range(n) ]
    sid=1
    for rm, n in [(ring(0,1/3),6),(ring(1/3,2/3),6),(ring(2/3,1.01),4)]:
        for sm in sectors(n):
            segs[sid] = rm & sm
            sid += 1
    return segs

def extract_features_from_pair(ed_bytes: bytes, es_bytes: bytes, spacing=(1.5,1.5,2.0)) -> tuple[dict[str,float], dict[int,float]]:
    ed = _load_mask_from_bytes(ed_bytes)
    es = _load_mask_from_bytes(es_bytes)

    lv_ed, rv_ed, my_ed = _volumes_mm3(ed, spacing)
    lv_es, rv_es, my_es = _volumes_mm3(es, spacing)
    lvef = (lv_ed - lv_es) / max(lv_ed, 1e-6)
    rvef = (rv_ed - rv_es) / max(rv_ed, 1e-6)
    # myocardium volume (mL) * density ~1.05 g/mL
    lvmass_g = (my_ed/1000.0) * 1.05

    # AHA16 thickness stats
    d,h,w = ed.shape
    bins = _aha16_bins(d,h,w)
    thk_ed = _wall_thickness(ed)
    thk_es = _wall_thickness(es)

    feat: dict[str,float] = {
        "LVEDV": lv_ed/1000.0, "LVESV": lv_es/1000.0, "LVEF": lvef,
        "RVEDV": rv_ed/1000.0, "RVESV": rv_es/1000.0, "RVEF": rvef,
        "LVMass": lvmass_g,
    }
    seg_scores: dict[int,float] = {}

    # within-case robust scaling for segment scores
    edvals, dvals = [], []
    for s, m in bins.items():
        v_ed = thk_ed[m & (ed==3)]
        v_es = thk_es[m & (es==3)]
        med_ed = float(np.median(v_ed)) if v_ed.size else 0.0
        med_es = float(np.median(v_es)) if v_es.size else 0.0
        feat[f"SEG{s}_thkED"] = med_ed
        feat[f"SEG{s}_thkES"] = med_es
        feat[f"SEG{s}_dThk"]  = med_es - med_ed
        edvals.append(med_ed); dvals.append(med_es - med_ed)

    edvals = np.asarray(edvals); dvals = np.asarray(dvals)
    med_ed, iqr_ed = np.median(edvals), (np.percentile(edvals,75)-np.percentile(edvals,25)+1e-6)
    med_d,  iqr_d  = np.median(dvals),  (np.percentile(dvals,75)-np.percentile(dvals,25)+1e-6)
    sigmoid = lambda z: 1/(1+np.exp(-z))
    for s in range(1,17):
        z1 = (feat[f"SEG{s}_thkED"] - med_ed)/iqr_ed
        z2 = (feat[f"SEG{s}_dThk"]  - med_d)/iqr_d
        score = 0.5*sigmoid(z1) + 0.5*(1.0 - sigmoid(z2))
        seg_scores[s] = float(np.clip(score,0,1))

    return feat, seg_scores
