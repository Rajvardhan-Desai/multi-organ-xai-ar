# backend/core/aha.py
from __future__ import annotations
import numpy as np
from typing import Dict, List

AHA16_NAMES = {
    1: "Basal Anterior", 2: "Basal Anteroseptal", 3: "Basal Inferoseptal",
    4: "Basal Inferior", 5: "Basal Inferolateral", 6: "Basal Anterolateral",
    7: "Mid Anterior", 8: "Mid Anteroseptal", 9: "Mid Inferoseptal",
    10: "Mid Inferior", 11: "Mid Inferolateral", 12: "Mid Anterolateral",
    13: "Apical Anterior", 14: "Apical Septal", 15: "Apical Inferior", 16: "Apical Lateral"
}

def aha16_bins(d:int, h:int, w:int):
    """
    Create intrinsic AHA bins using normalized long-axis (z in [0,1]) and in-plane angle.
    Returns a dict seg_id -> boolean mask [d,h,w].
    """
    z = np.linspace(0, 1, d).reshape(d,1,1).repeat(h,1).repeat(w,2)  # shape [d,h,w]
    yy = np.linspace(-1, 1, h).reshape(1,h,1).repeat(d,0).repeat(w,2)
    xx = np.linspace(-1, 1, w).reshape(1,1,w).repeat(d,0).repeat(h,1)
    theta = (np.arctan2(yy, xx) + np.pi) / (2*np.pi)  # [0,1)

    segs: Dict[int, np.ndarray] = {}
    def ring_mask(lo, hi): return (z>=lo) & (z<hi)
    def sectors(n):
        edges = np.linspace(0,1,n+1)
        return [ (theta>=edges[i]) & (theta<edges[i+1]) for i in range(n) ]

    # basal: 6, mid: 6, apical: 4
    ridx = 1
    for rm, n in [(ring_mask(0,1/3),6),(ring_mask(1/3,2/3),6),(ring_mask(2/3,1.01),4)]:
        for sm in sectors(n):
            segs[ridx] = (rm & sm)
            ridx += 1
    return segs

def aha16_names_map() -> Dict[int,str]:
    return dict(AHA16_NAMES)
