import os, tempfile
import nibabel as nib

def load_nii_bytes(data: bytes):
    # nibabel prefers a real file for gz; write to temp safely
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(data)
        tmp = f.name
    img = nib.load(tmp)
    try:
        os.remove(tmp)
    except Exception:
        pass
    return img
