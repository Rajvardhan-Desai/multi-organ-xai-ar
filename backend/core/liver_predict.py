# backend/core/liver_predict.py
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom


# ---------------------------------------------------------------------------
# Model: 3D ResNet with single-channel input (matches training notebook)
# ---------------------------------------------------------------------------

try:
    from torchvision.models.video import r3d_18
except Exception as e:
    raise RuntimeError(
        "torchvision.models.video.r3d_18 is required for liver fibrosis model."
    ) from e


class SqueezeExcite3D(nn.Module):
    """Minimal 3D squeeze-excitation block that matches the training checkpoint."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class ResNet3D_1ch(nn.Module):
    """
    Mirrors the training-time architecture:
    - torchvision r3d_18 backbone (3-channel stem)
    - SE heads after layers 2, 3, and 4
    - Dropout + Linear classification head
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        base = r3d_18(weights=None)
        self.stem = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.se2 = SqueezeExcite3D(128)
        self.se3 = SqueezeExcite3D(256)
        self.se4 = SqueezeExcite3D(512)

        in_f = base.fc.in_features
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_f, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.layer4(x)
        x = self.se4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Basic NIfTI helpers
# ---------------------------------------------------------------------------

def _load_nii(path: str | Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata()
    if data.ndim == 4:
        # take first channel / time frame if needed
        data = data[..., 0]
    return data.astype(np.float32)


def _resize_volume(vol: np.ndarray, out_shape=(64, 64, 64), order: int = 1) -> np.ndarray:
    """Resize 3D volume using scipy.ndimage.zoom."""
    vol = np.asarray(vol, np.float32)
    in_shape = vol.shape
    zoom_factors = [o / float(i) for o, i in zip(out_shape, in_shape)]
    return zoom(vol, zoom_factors, order=order)


def _zscore(vol: np.ndarray) -> np.ndarray:
    v = vol.astype(np.float32)
    m = float(v.mean())
    s = float(v.std() + 1e-6)
    return (v - m) / s


# ---------------------------------------------------------------------------
# Pseudo Couinaud segmentation + segment scoring (from notebook)
# ---------------------------------------------------------------------------

ROMAN = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
}


def _make_pseudo_couinaud_mask(liver_mask: np.ndarray) -> np.ndarray:
    """
    Very simplified pseudo-Couinaud segmentation:

    - Find bounding box of liver_mask > 0
    - Split into 2 x 2 x 2 bricks = 8 segments
    """
    liver_mask = (liver_mask > 0).astype(np.uint8)
    if liver_mask.sum() == 0:
        return np.zeros_like(liver_mask, dtype=np.int16)

    idx = np.argwhere(liver_mask > 0)
    z0, y0, x0 = idx.min(axis=0)
    z1, y1, x1 = idx.max(axis=0) + 1

    # midpoints
    zm = (z0 + z1) // 2
    ym = (y0 + y1) // 2
    xm = (x0 + x1) // 2

    seg = np.zeros_like(liver_mask, dtype=np.int16)

    # 8 segments (numbering chosen to be stable, not anatomical-absolute)
    # lower (inferior) half
    seg[z0:zm, y0:ym, x0:xm] = 1
    seg[z0:zm, y0:ym, xm:x1] = 2
    seg[z0:zm, ym:y1, x0:xm] = 3
    seg[z0:zm, ym:y1, xm:x1] = 4
    # upper (superior) half
    seg[zm:z1, y0:ym, x0:xm] = 5
    seg[zm:z1, y0:ym, xm:x1] = 6
    seg[zm:z1, ym:y1, x0:xm] = 7
    seg[zm:z1, ym:y1, xm:x1] = 8

    # only keep segments where liver_mask is present
    seg = seg * liver_mask
    return seg.astype(np.int16)


def _segment_scores_from_mask_array(cam: np.ndarray, segmask: np.ndarray) -> Dict[str, float]:
    """
    cam: 3D Grad-CAM map [D,H,W], non-negative
    segmask: int mask with values 1..8
    Returns { "I": score, "II": score, ... }
    """
    cam = np.asarray(cam, np.float32)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    out: Dict[str, float] = {}
    for sid in range(1, 9):
        roman = ROMAN[sid]
        region = cam[segmask == sid]
        if region.size == 0:
            out[roman] = 0.0
        else:
            # use mean activation as score (same spirit as notebook)
            out[roman] = float(region.mean())
    return out


# ---------------------------------------------------------------------------
# 3D Grad-CAM on ResNet3D_1ch
# ---------------------------------------------------------------------------

class LiverCamGrad:
    def __init__(self, net: nn.Module, layer_name: str = "layer4"):
        self.net = net.eval()
        self.layer_name = layer_name
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

    def _get_target_layer(self) -> nn.Module:
        layer = getattr(self.net, self.layer_name, None)
        if layer is None:
            raise RuntimeError(f"Layer {self.layer_name} not found in model.")
        return layer

    def _register_hooks(self):
        layer = self._get_target_layer()

        def fwd_hook(_m, _inp, out):
            self._acts = out.detach()

        def bwd_hook(_m, grad_in, grad_out):
            self._grads = grad_out[0].detach()

        layer.register_forward_hook(fwd_hook)
        layer.register_backward_hook(bwd_hook)

    def compute_cam(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        """
        x: [1,3,D,H,W]
        returns CAM resized back to volume size [D,H,W]
        """
        self._acts, self._grads = None, None
        self._register_hooks()

        x = x.requires_grad_(True)
        logits = self.net(x)  # [1, num_classes]
        logit = logits[0, target_class]
        self.net.zero_grad()
        logit.backward()

        if self._acts is None or self._grads is None:
            raise RuntimeError("Hooks did not capture activations / gradients.")

        acts = self._acts  # [1,C,d,h,w]
        grads = self._grads  # [1,C,d,h,w]

        # Global average pooling over spatial dims
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)  # [1,C,1,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=False)  # [1,d,h,w]
        cam = cam[0].detach().cpu().numpy()

        # ReLU and normalize
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to full volume size
        d, h, w = x.shape[2:]
        cam_resized = zoom(cam, (d / cam.shape[0], h / cam.shape[1], w / cam.shape[2]), order=1)
        return cam_resized.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level API used by FastAPI app
# ---------------------------------------------------------------------------

def _infer_on_nii(
    volume_path: str | Path,
    checkpoint_path: str,
    vol_size: Tuple[int, int, int] = (64, 64, 64),
    device: Optional[str] = None,
    want_xai: bool = True,
) -> Dict[str, Any]:
    """
    Main inference routine (mirrors your notebook).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    raw = _load_nii(volume_path)
    # Simple foreground mask (everything non-zero)
    liver_mask = (raw > 0).astype(np.uint8)

    vol = _resize_volume(liver_mask * raw, out_shape=vol_size, order=1)
    vol = _zscore(vol).astype(np.float32)

    x = torch.from_numpy(vol[None, None, ...])  # [1,1,D,H,W]
    # Training checkpoint expects 3-channel input, so tile the z-scored volume.
    x = x.repeat(1, 3, 1, 1, 1).to(device)  # [1,3,D,H,W]

    # Load model
    model = ResNet3D_1ch(num_classes=2).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(x)
        proba = F.softmax(logits, dim=1)[0].cpu().numpy()

    label_map = {0: "Healthy", 1: "Cirrhosis"}
    pred_idx = int(np.argmax(proba))
    pred_label = label_map[pred_idx]
    proba_map = {label_map[i]: float(proba[i]) for i in range(len(label_map))}

    result: Dict[str, Any] = {
        "prediction": pred_label,
        "proba": proba_map,
    }

    if want_xai:
        # Grad-CAM
        cam_engine = LiverCamGrad(model)
        cam = cam_engine.compute_cam(x, target_class=pred_idx)

        # Pseudo-Couinaud segmentation in model space
        seg_mask = _make_pseudo_couinaud_mask((vol > 0).astype(np.float32))
        seg_scores = _segment_scores_from_mask_array(cam, seg_mask)

        # derive simple hotspot list: top 3 segments
        sorted_segs = sorted(seg_scores.items(), key=lambda kv: kv[1], reverse=True)
        hotspots = [
            {"segment": roman, "score": float(score)}
            for roman, score in sorted_segs[:3]
        ]

        result["segment_scores"] = seg_scores
        result["xai"] = {
            "method": "gradcam3d",
            "top_regions": [
                {
                    "label_id": i + 1,
                    "label_name": f"Segment {roman}",
                    "score": float(score),
                }
                for i, (roman, score) in enumerate(sorted_segs[:8])
            ],
            "hotspots": hotspots,
        }

    return result


def predict_liver_fibrosis(model_ckpt_path: str, volume_bytes: bytes, want_xai: bool) -> Dict[str, Any]:
    """
    FastAPI-friendly wrapper: takes NIfTI bytes, writes a temp file,
    runs _infer_on_nii, returns JSON-serializable dict.
    """
    if not volume_bytes:
        raise ValueError("Empty NIfTI upload for liver fibrosis.")

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(volume_bytes)
        tmp_path = tmp.name

    try:
        out = _infer_on_nii(
            volume_path=tmp_path,
            checkpoint_path=model_ckpt_path,
            want_xai=want_xai,
        )
        return out
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
