# EX-AI-AR Project Guide

## 1. Introduction
EX-AI-AR couples a FastAPI backend with a Next.js frontend to deliver explainable brain-disease inference using MALPEM segmentations. Users upload `.nii.gz` files, receive classifier outputs plus top contributing ROIs, and view them on interactive 3D/AR brain meshes.

---

## 2. Repository Layout

```
backend/
  app.py                  # FastAPI entry point
  core/                   # ML + data utilities
  models/...              # Saved ML artifacts (joblib, LUTs)
  static/...              # GLB assets served by backend
  requirements.txt        # Backend Python deps

frontend/
  app/                    # Next.js app router pages
  components/             # UI + visualization widgets
  public/static/brain/    # Client-side GLB + mappings
  package.json            # Frontend npm manifest
```

Ignored paths (node_modules, build artifacts, env files, caches, etc.) are declared in `.gitignore`.

---

## 3. Data & Asset Inventory

| Artifact | Purpose | Location |
| --- | --- | --- |
| `model.joblib`, `cn_reference.joblib` | Trained scikit-learn estimator bundle plus optional control-group reference | `backend/models/<organ>/<disease>/` |
| `lut_parsed.csv` | Label ? brain-region name mapping consumed by feature extraction | same as above |
| `mapping_lut_to_glb.csv` | Label IDs to GLB nodes for highlight alignment | same as above and mirrored under `frontend/public/static/brain/` |
| `brain.glb` | 3D mesh used by both React Three Fiber and `<model-viewer>` components | `backend/static/brain/brain.glb` & `frontend/public/static/brain/brain.glb` |
| `mapping.json` | Rich mapping for `<model-viewer>` (may include multiple mesh targets per label) | `frontend/public/static/brain/mapping.json` |

---

## 4. Backend Architecture

### 4.1 App bootstrap (`backend/app.py`)
- Scans `models/<organ>/<disease>` for `model.joblib` + `lut_parsed.csv`, registers optional `meta.json` and `cn_reference.joblib` (`backend/app.py:33`).
- Preloads LUTs via `core.lut.load_lut` and model bundles via `core.models.load_model_bundle`, caching everything in memory (`backend/app.py:63`).
- Allows custom class-name mappings defined in `meta.json` under `class_name_map` (`backend/app.py:71`).

### 4.2 REST API
- `GET /registry`: Enumerates organs and diseases discovered at startup so the frontend can populate selectors (`backend/app.py:91`).
- `POST /infer`: Accepts `organ`, `disease`, `xai` query params plus a `.nii.gz` upload; runs inference and returns prediction payloads (`backend/app.py:103`).

### 4.3 Prediction workflow (`backend/core/predict.py`)
1. `extract_roi_features` converts the segmentation into ROI volumes, returning both a NumPy array and intracranial volume (ICV) surrogate (`backend/core/features.py:8`).
2. Features are aligned to the models `x_cols`; missing columns default to zero.
3. Class probability handling:
   - Uses `predict_proba` when available (`backend/core/predict.py:101`).
   - Falls back to deterministic predictions when only `predict` exists.
4. Label normalization:
   - Applies `class_name_map` when provided (`backend/core/predict.py:118`).
   - Falls back to standard CN/AD naming when possible (`backend/core/predict.py:123`).
5. Explainability:
   - Uses `feature_importances_` or `coef_` when dimensions align (`backend/core/predict.py:128`).
   - Otherwise ranks regions by normalized volume as a robust fallback (`backend/core/predict.py:142`).
6. Response payload includes prediction, probability map, ICV, used feature columns, top regions, and optional XAI metadata.

### 4.4 Supporting utilities
- `core/model_io.py`: Safely unwraps estimators regardless of how the joblib bundle was saved (`backend/core/model_io.py:5`).
- `core/registry.py`: Discovers required artifacts when pointed at an arbitrary model directory (`backend/core/registry.py:5`).
- `utils_io.py` and `core/io_utils.py`: Load `.nii.gz` data from bytes or zip members via temp files to satisfy nibabels requirements (`backend/utils_io.py:5`, `backend/core/io_utils.py:1`).

---

## 5. Frontend Architecture

### 5.1 App router page (`frontend/app/page.tsx`)
- Loads available organs/diseases from `/registry`, falling back to `{ brain: ["alzheimer"] }` if the backend is unreachable (`frontend/app/page.tsx:33`).
- Maintains state for selected organ/disease, uploaded file, XAI toggle, and inference results.
- `onInfer` builds a `FormData` request and fetches `/infer`, surfacing JSON directly for transparency (`frontend/app/page.tsx:77`).
- Highlights top regions in an embedded viewer and prints raw JSON + list view for debugging (`frontend/app/page.tsx:179`).

### 5.2 Visualization components
- `BrainMV`: Lazy-loads `@google/model-viewer`, fetches `mapping.json`, fades all materials, then re-highlights those tied to the top regions (`frontend/components/BrainMV.tsx:26`, `frontend/components/BrainMV.tsx:70`).
- `BrainViewer`: Alternative React Three Fiber viewer that traverses the GLB scene and recolors meshes using emissive highlights (`frontend/components/BrainViewer.tsx:20`, `frontend/components/BrainViewer.tsx:42`).
- `BarChart`: Simple horizontal bar visualization for ROI contributions (currently unused in the page but available) (`frontend/components/BarChart.tsx:4`).

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

**Body**
- multipart/form-data with `file` containing a `.nii` or `.nii.gz` MALPEM segmentation.

**Success response**
```json
{
  "prediction": "AD",
  "proba": { "CN": 0.12, "AD": 0.88 },
  "icv_mm3": 1500000,
  "used_features": ["vol_1", "..."],
  "top_regions": [{ "label_id": 6, "label_name": "Left Amygdala", "score": 0.34 }, ...],
  "xai": { "method": "feature_importance", "top_regions": [...] }
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
- Copy trained artifacts into `backend/models/<organ>/<disease>/` with at least `model.joblib` and `lut_parsed.csv`.
- Optional files: `meta.json` (with `class_name_map`, preprocessing hints), `cn_reference.joblib`, `shap_background.npy`, `preproc.json`.
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
- Restrict allowed origins in FastAPI `CORSMiddleware` (`backend/app.py:17`) for production.
- Validate uploads (size limits, MIME) before storing; optionally add antivirus or sandbox scanning.

---

## 9. Extensibility

1. **New organs/diseases**: drop additional directories in `backend/models/<organ>/<disease>`, add associated LUT/mapping files, restart backend.
2. **Custom explainability**: augment `predict` to call SHAP, LIME, or Grad-CAM; store background datasets (e.g., `shap_background.npy`) noted in `registry.yaml`.
3. **Alternative viewers**: expose `BrainViewer` on the frontend to offer a React Three Fiber mode; reuse existing `weightsByLabel` API.
4. **UI polish**: integrate `BarChart` next to JSON to highlight top ROIs with bars instead of raw text.
5. **Preprocessing**: describe pre-processing steps (normalization, smoothing) in `preproc.json` and teach the backend to apply them before inference.

---

## 10. Troubleshooting

| Symptom | Likely Cause | Remedy |
| --- | --- | --- |
| `/registry` returns `{ "organs": [] }` | No valid model directories or missing `model.joblib`/`lut_parsed.csv` | Confirm directory structure and filenames, then restart backend (`backend/app.py:33`) |
| `/infer` responds 404 Model not found | Organ/disease params mismatch exactly loaded keys | Check casing and spelling; inspect server logs printed during `_scan_models` |
| `/infer` 500 Inference error: ... nibabel... | Uploaded file not NIfTI or corrupt | Validate local `.nii.gz` and ensure MALPEM label set matches LUT |
| Viewer shows no highlights | `mapping.json` or `mapping_lut_to_glb.csv` lacking entries for top label IDs | Extend mappings; check console warnings from `BrainMV` (`frontend/components/BrainMV.tsx:130`) |
| CLIs claim `guide.md` missing | File not created (especially in read-only setups) | Create locally before running `pandoc` |

---

## 11. Useful Commands

- List tracked files (respects `.gitignore`): `rg --files`
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
