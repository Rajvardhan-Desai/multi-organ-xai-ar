from __future__ import annotations
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import nibabel as nib
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Existing brain helpers
from core.lut import load_lut               # returns {int_id: "label name"}
from core.models import load_model_bundle   # returns {"model","x_cols","classes",...}
from core.predict import predict            # brain path: uses class_name_map & xai

# New heart predictor
from core.heart_predict import predict_heart_cardio

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
STATIC_DIR = BASE_DIR / "static"

# -----------------------------------------------------------------------------
# Registry bootstrap: scan models/<organ>/<disease>
# - model.joblib is required
# - lut_parsed.csv is optional (brain needs it; heart does not)
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
            if not model_p.exists():
                continue
            key = (organ_dir.name, disease_dir.name)
            item = {
                "model": model_p,
                "model_dir": disease_dir,  # for heart path artifacts (scaler/x_cols/label map)
            }
            lut_p = disease_dir / "lut_parsed.csv"
            if lut_p.exists():
                item["lut"] = lut_p
            item["meta"] = disease_dir / "meta.json"             # optional
            item["cnref"] = disease_dir / "cn_reference.joblib"  # optional
            registry[key] = item

            print("[models] probing:", str(disease_dir))
            print("         model:", str(model_p))
            if lut_p.exists():
                print("         lut  :", str(lut_p))
    return registry

PATHS = _scan_models(MODELS_DIR)

# Preload bundles & LUTs in memory
MODELS: Dict[Tuple[str, str], Dict[str, Any]] = {}
LUTS: Dict[Tuple[str, str], Dict[int, str]] = {}

for key, paths in PATHS.items():
    organ, disease = key

    # Load model bundle (works for both brain and heart)
    bundle = load_model_bundle(str(paths["model"]))
    bundle["_model_dir"] = str(paths["model_dir"])  # keep path for heart artifacts

    # Optional class-name map via meta.json
    class_map = {"0": "CN", "1": "AD", "CN": "CN", "AD": "AD"}  # default for brain
    meta_p: Optional[Path] = paths.get("meta")  # type: ignore
    if meta_p and meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
            if isinstance(meta.get("class_name_map"), dict):
                class_map.update({str(k): str(v) for k, v in meta["class_name_map"].items()})
            bundle["_meta"] = meta
        except Exception:
            pass
    bundle["_class_name_map"] = class_map

    # Load LUT only if present (brain)
    lut_p: Optional[Path] = paths.get("lut")  # type: ignore
    if lut_p and lut_p.exists():
        lut = load_lut(str(lut_p))
        LUTS[key] = lut
        print(f"[lut] {organ}/{disease}: loaded {len(lut)} labels")
    else:
        print(f"[lut] {organ}/{disease}: no LUT (ok for heart)")

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
    org_to_dis: Dict[str, List[str]] = {}
    for (organ, disease) in MODELS.keys():
        org_to_dis.setdefault(organ, []).append(disease)
    organs_list = [{"organ": org, "diseases": sorted(dis)} for org, dis in sorted(org_to_dis.items())]
    return {"organs": organs_list}

@app.post("/infer")
async def infer(
    organ: str = Query(...),
    disease: str = Query(...),
    # brain: single file
    file: UploadFile | None = File(default=None),
    # heart: ED/ES masks (preferred)
    ed_file: UploadFile | None = File(default=None),
    es_file: UploadFile | None = File(default=None),
    xai: bool = Query(False),
):
    key = (organ, disease)
    if key not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found for organ/disease")

    # -------- HEART: cardiomyopathy (expects masks; LUT not needed) --------
    if organ.lower() == "heart" and disease.lower() == "cardiomyopathy":
        # Accept ED/ES separately; allow fallback: single file used as both
        if ed_file is None and file is None:
            raise HTTPException(400, "Provide ed_file and es_file or a single 'file' for fallback.")
        try:
            if ed_file is not None:
                ed_bytes = await ed_file.read()
                es_bytes = await (es_file.read() if es_file is not None else ed_file.read())
            else:
                # fallback: 'file' is reused for ES
                fb = await file.read()  # type: ignore[arg-type]
                ed_bytes = fb
                es_bytes = fb

            model_dir = MODELS[key]["_model_dir"]
            payload = predict_heart_cardio(model_dir, ed_bytes, es_bytes, want_xai=xai)
            return payload
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"Inference error: {e}"})

    # -------- BRAIN (legacy path): needs LUT + single file --------
    if file is None:
        raise HTTPException(400, "Provide 'file' (.nii/.nii.gz) for brain inference.")
    if key not in LUTS:
        raise HTTPException(500, "LUT not loaded for this brain model.")
    try:
      contents = await file.read()
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
