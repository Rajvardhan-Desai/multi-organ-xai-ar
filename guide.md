# EX-AI-AR Project Guide

## 1. Introduction
EX-AI-AR couples a FastAPI backend with a Next.js frontend to deliver explainable multi‑organ inference:
- Brain (Alzheimer): MALPEM segmentation upload, classifier outputs, and top contributing ROIs highlighted on a 3D/AR brain.
- Heart (Cardiomyopathy): ED/ES myocardial masks, classifier outputs and AHA16 segment scores highlighted on a 3D heart.

Users upload `.nii/.nii.gz` files, receive predictions with explainability, and visualize results on interactive 3D/AR meshes.

---

## 2. Repository Layout

```
backend/
  app.py                      # FastAPI entry point
  core/                       # ML + data utilities
    heart_features.py         # Heart ED/ES feature extraction + AHA16
    heart_predict.py          # Heart cardiomyopathy predictor
  models/...                  # Saved ML artifacts (joblib, LUTs)
  static/
    brain/brain.glb           # Server-side brain mesh
    heart/heart.glb           # Server-side heart mesh
  requirements.txt            # Backend Python deps

frontend/
  app/                        # Next.js app router pages
  components/                 # UI + visualization widgets
    BrainMV.tsx               # Brain <model-viewer> highlighter
    HeartMV.tsx               # Heart R3F highlighter
  public/static/brain/        # Client-side brain GLB + mapping
  public/static/heart/        # Client-side heart GLB + mapping + legend
  package.json                # Frontend npm manifest
```

Ignored paths (node_modules, build artifacts, env files, caches, etc.) are declared in `.gitignore`.

---

## 3. Data & Asset Inventory

| Artifact | Purpose | Location |
| --- | --- | --- |
| `model.joblib`, `cn_reference.joblib` | Trained scikit-learn estimator bundle plus optional control-group reference | `backend/models/<organ>/<disease>/` |
| `lut_parsed.csv` | Label ? brain-region name mapping consumed by feature extraction | same as above |
| `mapping_lut_to_glb.csv` | Brain: label IDs to GLB nodes for highlight alignment | same as above and mirrored under `frontend/public/static/brain/` |
| `brain.glb` | Brain 3D mesh for `<model-viewer>` and R3F | `backend/static/brain/brain.glb` & `frontend/public/static/brain/brain.glb` |
| `mapping.json` | Brain: rich mapping for `<model-viewer>` targets per label | `frontend/public/static/brain/mapping.json` |
| `heart.glb` | Heart 3D mesh for R3F viewer | `backend/static/heart/heart.glb` & `frontend/public/static/heart/heart.glb` |
| `mapping.json` | Heart: mapping AHA16 segments → GLB node names | `frontend/public/static/heart/mapping.json` |
| `legend.json` | Heart: display names for AHA16 segments | `frontend/public/static/heart/legend.json` |

---

## 4. Backend Architecture

### 4.1 App bootstrap (`backend/app.py`)
- Scans `models/<organ>/<disease>` for `model.joblib`, registers optional `lut_parsed.csv` (brain), `meta.json`, `cn_reference.joblib` (`backend/app.py:39`).
- Preloads model bundles and any LUTs into memory (`backend/app.py:76`).
- Supports custom class-name mappings via `meta.json.class_name_map` (`backend/app.py:84`).

### 4.2 REST API
- `GET /registry`: Enumerates organs and diseases discovered at startup so the frontend can populate selectors (`backend/app.py:115`).
- `POST /infer`: Core inference endpoint (`backend/app.py:125`).
  - Brain: supply `file` (.nii/.nii.gz segmentation). Requires LUT for the selected model.
  - Heart (cardiomyopathy): prefer `ed_file` and `es_file` (.nii/.nii.gz masks). If only one file is provided, pass it as `file` and it is reused for both ED/ES.

### 4.3 Prediction workflow — brain (`backend/core/predict.py`)
1. `extract_roi_features` returns ROI volumes and an intracranial volume (ICV) surrogate (`backend/core/features.py:8`).
2. Features align to the model’s `x_cols`; missing columns default to zero.
3. Probabilities: use `predict_proba` when available (`backend/core/predict.py:99`), otherwise fabricate a deterministic map from `predict`.
4. Label normalization via `class_name_map` when present (`backend/core/predict.py:112`).
5. Explainability: use `feature_importances_`/`coef_` if shapes match (`backend/core/predict.py:128`), else fallback to normalized volume ranking (`backend/core/predict.py:142`).
6. Response includes prediction, probability map, ICV, used features, top regions, and optional XAI metadata.

### 4.4 Prediction workflow — heart (`backend/core/heart_predict.py`)
- Loads `model.joblib`, `scaler.joblib`, `x_cols.json`, and `xgb_label_map.json` from the model directory.
- Extracts features from ED/ES masks (volumes, EF, myocardium mass, AHA16 thickness stats) and derives per‑segment scores (`backend/core/heart_features.py`).
- Predicts robustly across xgboost versions, normalizes probabilities, and returns:
  - `prediction`, `proba`, `used_features`, `segment_scores`, and optional XAI via Booster gain (`backend/core/heart_predict.py:78`).

### 4.4 Supporting utilities
- `core/model_io.py`: Safely unwraps estimators regardless of how the joblib bundle was saved (`backend/core/model_io.py:5`).
- `core/registry.py`: Discovers required artifacts when pointed at an arbitrary model directory (`backend/core/registry.py:5`).
- `core/mapping.py`: Helpers for reading LUT and LUT→GLB mapping CSVs (`backend/core/mapping.py:1`).
- `core/aha.py`: AHA16 segment names and intrinsic binning helpers (not directly wired into the current heart pipeline) (`backend/core/aha.py:1`).
- `utils_io.py` and `core/io_utils.py`: Load `.nii.gz` data from bytes or zip members via temp files to satisfy nibabel’s requirements (`backend/utils_io.py:1`, `backend/core/io_utils.py:1`).

---

## 5. Frontend Architecture

### 5.1 App router page (`frontend/app/page.tsx`)
- Loads available organs/diseases from `/registry`, falling back to brain+heart defaults if unreachable (`frontend/app/page.tsx:37`).
- Maintains state for selected organ/disease, uploaded files (brain: one; heart: ED/ES), XAI toggle, and results.
- `onInfer` builds a `FormData` request per organ and calls `/infer` (`frontend/app/page.tsx:92`).
- Renders `HeartMV` for heart with `segment_scores`, or `BrainMV` for brain with top regions; prints raw JSON for debugging (`frontend/app/page.tsx:257`).

### 5.2 Visualization components
- `BrainMV`: Lazy-loads `@google/model-viewer`, fetches `mapping.json`, fades all materials, then re-highlights those tied to top regions (`frontend/components/BrainMV.tsx:26`, `frontend/components/BrainMV.tsx:70`).
- `HeartMV`: React Three Fiber viewer; loads heart GLB and mapping, fades everything, then highlights nodes mapped from AHA16 segment scores; includes a debug button to dump GLB node names (`frontend/components/HeartMV.tsx:20`, `frontend/app/page.tsx:183`).
- `BrainViewer`: Alternative R3F brain viewer (not currently mounted in page) (`frontend/components/BrainViewer.tsx:20`).
- `BarChart`: Simple horizontal bar visualization for ROI contributions (currently unused) (`frontend/components/BarChart.tsx:4`).

### 5.3 Styling and layout
- `globals.css` defines basic tokens plus Tailwind-like utility classes to keep JSX concise without installing Tailwind (`frontend/app/globals.css:1`).
- `app/layout.tsx` sets global metadata and wraps pages in a minimal HTML scaffold (`frontend/app/layout.tsx:2`).

---

## 6. API Contract

### 6.1 `GET /registry`
**Response**
```json
{
  "organs": [
    { "organ": "brain", "diseases": ["alzheimer"] }
  ]
}
```
(Empty array if no models are loaded.)

### 6.2 `POST /infer`
**Query params**
- `organ` (string, required)
- `disease` (string, required)
- `xai` (bool/int, optional; default `false`)

**Body** (multipart/form-data)
- Brain: `file` — MALPEM `.nii/.nii.gz` segmentation.
- Heart: `ed_file` and `es_file` — `.nii/.nii.gz` masks. Fallback: single `file` used as both.

**Success response (brain)**
```json
{
  "prediction": "AD",
  "proba": { "CN": 0.12, "AD": 0.88 },
  "icv_mm3": 1500000,
  "used_features": ["vol_1", "..."],
  "top_regions": [{ "label_id": 6, "label_name": "Left Amygdala", "score": 0.34 }],
  "xai": { "method": "feature_importance", "top_regions": [] }
}
```

**Success response (heart)**
```json
{
  "prediction": "DCM",
  "proba": { "HCM": 0.15, "DCM": 0.85 },
  "used_features": ["LVEDV", "SEG1_thkED", "..."],
  "segment_scores": { "1": 0.71, "2": 0.63, "...": 0.12 },
  "xai": { "method": "gain", "top_regions": [{ "label_id": 14, "label_name": "AHA14", "score": 0.12 }] }
}
```

**Error responses**
- 404 if organ/disease not registered.
- 500 with `"detail": "Inference error: ..."` for corrupted uploads or execution faults.

---

## 7. Setup & Usage

### 7.1 Backend
1. `cd backend`
2. `python -m venv .venv && .\\.venv\\Scripts\\activate` (Windows) or `source .venv/bin/activate` (Unix).
3. `pip install -r requirements.txt`.
4. Launch: `uvicorn app:app --reload --host 0.0.0.0 --port 8000`.

**Model onboarding**
- Brain: copy `model.joblib` and `lut_parsed.csv` into `backend/models/<organ>/<disease>/`.
- Heart: copy `model.joblib`, `scaler.joblib`, `x_cols.json`, `xgb_label_map.json` into `backend/models/heart/cardiomyopathy/`.
- Optional: `meta.json` (`class_name_map`), `cn_reference.joblib`, `shap_background.npy`, `preproc.json` (currently informational; `backend/registry.yaml` is not consumed by the app).
- Restart the backend to pick up new directories.

### 7.2 Frontend
1. `cd frontend`
2. `npm install`
3. `npm run dev` (defaults to http://localhost:3000).
4. Ensure `NEXT_PUBLIC_BACKEND` points to the running backend, e.g.:
   - PowerShell: `$env:NEXT_PUBLIC_BACKEND="http://127.0.0.1:8000"; npm run dev`
   - Unix: `NEXT_PUBLIC_BACKEND=http://127.0.0.1:8000 npm run dev`

### 7.3 End-to-end flow
1. Start backend and frontend dev servers.
2. Open the frontend in a browser.
3. Select organ/disease (auto-populated).
4. Upload a MALPEM `.nii.gz` file.
5. Toggle XAI if desired and click Run Inference.
6. Review JSON output, top regions list, and highlighted 3D brain.

---

## 8. Deployment Guidance

### 8.1 Backend
- Use a production ASGI server (e.g., `uvicorn --workers 4`, Gunicorn with `uvicorn.workers.UvicornWorker`, or Hypercorn).
- Mount static routes (GLB assets) through FastAPI or front them via CDN.
- Configure environment variables or CLI flags for bind host/port, logging, etc.
- Consider async file storage (S3, Azure Blob, GCS) if model artifacts are large.

### 8.2 Frontend
- `npm run build` produces a `.next` production build.
- Deploy via Vercel, Netlify, or any Node hosting. Ensure `NEXT_PUBLIC_BACKEND` is set in deployment environment variables.
- Serve `/public/static/brain/*` as static assets so GLB and mappings load without CORS issues.

### 8.3 Security & Compliance
- Enforce HTTPS.
- Restrict allowed origins in FastAPI `CORSMiddleware` for production (`backend/app.py:21`).
- Validate uploads (size limits, MIME) before storing; optionally add antivirus or sandbox scanning.

---

## 9. Extensibility

1. **New organs/diseases**: drop additional directories in `backend/models/<organ>/<disease>`, add associated LUT/mapping files, restart backend.
2. **Custom explainability**: augment `predict` to call SHAP, LIME, or Grad-CAM; store background datasets (e.g., `shap_background.npy`) noted in `registry.yaml`.
3. **Alternative viewers**: expose `BrainViewer` on the frontend to offer a React Three Fiber mode; reuse existing `weightsByLabel` API. `HeartMV` is already R3F-based.
4. **UI polish**: integrate `BarChart` next to JSON to highlight top ROIs with bars instead of raw text.
5. **Preprocessing**: describe pre-processing steps (normalization, smoothing) in `preproc.json` and teach the backend to apply them before inference.

---

## 10. Troubleshooting

| Symptom | Likely Cause | Remedy |
| --- | --- | --- |
| `/registry` returns `{ "organs": [] }` | No valid model directories or missing `model.joblib`/`lut_parsed.csv` | Confirm directory structure and filenames, then restart backend (`backend/app.py:33`) |
| `/infer` responds 404 Model not found | Organ/disease params mismatch exactly loaded keys | Check casing and spelling; inspect server logs printed during `_scan_models` |
| `/infer` 500 Inference error: ... nibabel... | Uploaded file not NIfTI or corrupt | Validate local `.nii.gz` and ensure MALPEM label set matches LUT |
| Viewer shows no highlights (brain) | Brain mapping missing entries | Extend brain `mapping.json`; check console warnings from `BrainMV` (`frontend/components/BrainMV.tsx:141`) |
| Viewer shows no highlights (heart) | Heart mapping missing entries or mismatched node names | Extend heart `mapping.json`; use the page “Debug: Dump GLB names” button to inspect names (`frontend/app/page.tsx:183`) |
| Heart 400 “Provide ed_file and es_file...” | Missing ED/ES uploads | Upload both masks, or provide a single file in `file` which is reused for ED/ES |
| CLIs claim `guide.md` missing | File not created (especially in read-only setups) | Create locally before running `pandoc` |

---

## 11. Useful Commands

- List tracked files (respects `.gitignore`): `git ls-files` (or `rg --files` if available)
- Inspect NIfTI metadata quickly:
  ```python
  python - <<'PY'
  import nibabel as nib, sys
  img = nib.load(sys.argv[1])
  print(img.shape, img.header.get_zooms())
  PY path/to/file.nii.gz
  ```
- Generate guide PDF once `guide.md` exists: `pandoc guide.md -o guide.pdf`
- Start backend with logs: `uvicorn app:app --reload --log-level info`
- Build frontend for production: `npm run build && npm run start`

---

## 12. Appendix: Key File References

- FastAPI service: `backend/app.py`
- Prediction pipeline: `backend/core/predict.py`
- Feature extraction: `backend/core/features.py`
- LUT loader: `backend/core/lut.py`
- Model loader: `backend/core/models.py`
- Frontend page: `frontend/app/page.tsx`
- Model-viewer component: `frontend/components/BrainMV.tsx`
- Three.js viewer: `frontend/components/BrainViewer.tsx`
- Brain assets: `frontend/public/static/brain/*`, `backend/static/brain/brain.glb`
