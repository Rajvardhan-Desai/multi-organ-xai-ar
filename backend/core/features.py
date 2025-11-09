# core/features.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Literal

import numpy as np
import nibabel as nib

def extract_roi_features(
    seg_img: nib.Nifti1Image,
    lut: Dict[int, str],
    return_type: Literal["array", "dict"] = "array",
) -> Tuple[np.ndarray | Dict[str, float], float]:
    """
    Compute ROI volumes (mm^3) per label id present in `lut`.
    Returns (features, icv_mm3)

    - If return_type="array": returns a numpy array shaped (N_labels,)
      ordered by label_id ascending (1..138 typically).
      The column names would be ["vol_1","vol_2",...].
    - If return_type="dict": returns {"vol_<id>": value_mm3, ...}

    ICV (intracranial volume proxy) is sum of all ROI volumes in mm^3.
    """
    arr = seg_img.get_fdata().astype(np.int32)
    vx = seg_img.header.get_zooms()
    vox_mm3 = float(vx[0] * vx[1] * vx[2])

    label_ids: List[int] = sorted(lut.keys())
    vols = []
    vol_dict: Dict[str, float] = {}
    total = 0.0

    for lid in label_ids:
        cnt = int(np.count_nonzero(arr == lid))
        vol = cnt * vox_mm3
        vols.append(vol)
        vol_dict[f"vol_{lid}"] = vol
        total += vol

    if return_type == "dict":
        return vol_dict, total
    else:
        return np.asarray(vols, dtype=np.float64), total
