import pandas as pd
from typing import Dict, List

def read_lut_csv(path: str) -> Dict[int, str]:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # Expect label_id,label_name
    assert "label_id" in df.columns and "label_name" in df.columns, "lut.csv must have label_id,label_name"
    return {int(r["label_id"]): str(r["label_name"]) for _, r in df.iterrows()}

def read_lut_to_glb_csv(path: str) -> Dict[int, List[str]]:
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    assert "label_id" in df.columns and "glb_node" in df.columns, "mapping_lut_to_glb.csv must have label_id, glb_node"
    out: Dict[int, List[str]] = {}
    for _, r in df.iterrows():
        lid = int(r["label_id"])
        node = str(r["glb_node"]).strip()
        if node:
            out.setdefault(lid, []).append(node)
    return out
