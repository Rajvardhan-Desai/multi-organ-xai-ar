import os, io, json, zipfile, tempfile
from typing import Optional
import nibabel as nib

def nii_from_bytes_or_path(data: bytes=None, path: Optional[str]=None):
    """
    Robustly load NIfTI from bytes or path. Writes to temp if needed.
    """
    if path:
        return nib.load(path)
    if data is None:
        raise ValueError("No data/path for NIfTI")
    # Write to a temp .nii.gz to ensure nibabel works on all platforms
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(data)
        tmp = f.name
    img = nib.load(tmp)
    try:
        os.remove(tmp)
    except Exception:
        pass
    return img

def read_nii_from_zip_member(zip_path: str, member: str):
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            data = fh.read()
    return nii_from_bytes_or_path(data=data)

def json_dump(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def json_load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
