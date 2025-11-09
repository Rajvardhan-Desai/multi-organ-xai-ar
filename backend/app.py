# app.py
from __future__ import annotations
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, List

import nibabel as nib
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.lut import load_lut              # returns {int_id: "label name"}
from core.models import load_model_bundle  # returns {"model","x_cols","classes",...}
from core.predict import predict           # uses class_name_map & xai

app = FastAPI(title="EX-AI-AR Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# -----------------------------------------------------------------------------
# Registry bootstrap: scan models/<organ>/<disease> for model.joblib & lut_parsed.csv
# -----------------------------------------------------------------------------
def _scan_models(root: Path) -> Dict[Tuple[str, str], Dict[str, Path]]:
    registry: Dict[Tuple[str, str], Dict[str, Path]] = {}
    if not root.exists():
        return registry
    for organ_dir in root.iterdir():
        if not organ_dir.is_dir():
            continue
        for disease_dir in organ_dir.iterdir():
            if not disease_dir.is_dir():
                continue
            model_p = disease_dir / "model.joblib"
            lut_p   = disease_dir / "lut_parsed.csv"
            if model_p.exists() and lut_p.exists():
                key = (organ_dir.name, disease_dir.name)
                registry[key] = {
                    "model": model_p,
                    "lut": lut_p,
                    "meta": disease_dir / "meta.json",           # optional
                    "cnref": disease_dir / "cn_reference.joblib" # optional
                }
                print("[models] probing:", str(disease_dir))
                print("         model:", str(model_p))
                print("         lut  :", str(lut_p))
    return registry

PATHS = _scan_models(MODELS_DIR)

# Preload bundles & LUTs in memory
MODELS: Dict[Tuple[str, str], Dict[str, Any]] = {}
LUTS: Dict[Tuple[str, str], Dict[int, str]] = {}

for key, paths in PATHS.items():
    organ, disease = key
    lut = load_lut(str(paths["lut"]))
    LUTS[key] = lut
    bundle = load_model_bundle(str(paths["model"]))
    # Optional friendly-class mapping per model (can be overridden via meta.json)
    class_map = {"0": "CN", "1": "AD", "CN": "CN", "AD": "AD"}
    meta_p = paths.get("meta")
    if meta_p and meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
            if isinstance(meta.get("class_name_map"), dict):
                class_map.update({str(k): str(v) for k, v in meta["class_name_map"].items()})
        except Exception:
            pass
    bundle["_class_name_map"] = class_map

    print(f"[lut] {organ}/{disease}: loaded {len(lut)} labels")
    est = bundle["model"]
    has_proba = hasattr(est, "predict_proba")
    print(f"[models] {organ}/{disease}: {est.__class__.__name__} "
          f"(predict_proba={has_proba}), keys={list(bundle.keys())}")
    MODELS[key] = bundle

# -----------------------------------------------------------------------------
# Public endpoints
# -----------------------------------------------------------------------------
@app.get("/registry")
def registry():
    if not MODELS:
        return {"organs": []}
    # Build organs -> diseases list
    org_to_dis: Dict[str, List[str]] = {}
    for (organ, disease) in MODELS.keys():
        org_to_dis.setdefault(organ, []).append(disease)
    # Sort for stable UI
    organs_list = [{"organ": org, "diseases": sorted(dis)} for org, dis in sorted(org_to_dis.items())]
    return {"organs": organs_list}

@app.post("/infer")
async def infer(
    organ: str = Query(...),
    disease: str = Query(...),
    file: UploadFile = File(...),
    xai: bool = Query(False),
):
    key = (organ, disease)
    if key not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found for organ/disease")

    contents = await file.read()
    try:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            img = nib.load(tmp_path)
            bundle = MODELS[key]
            lut = LUTS[key]
            result = predict(
                bundle=bundle,
                seg_img=img,
                lut=lut,
                produce_xai=xai,
                class_name_map=bundle.get("_class_name_map"),
            )
            return result
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Inference error: {e}"})
