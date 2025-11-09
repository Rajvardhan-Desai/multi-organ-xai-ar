# core/lut.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

import csv

def load_lut(path: str) -> Dict[int, str]:
    """
    Load LUT in one of two formats and always return {label_id: label_name}.

    Supported:
      1) CSV with headers: label_id,label_name
      2) Raw text (MALPEM style) lines: "<id> R G B 1 1 <name...>"
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LUT not found: {p}")

    # Try CSV first
    try:
        with p.open("r", newline="", encoding="utf-8") as f:
            sniffer = csv.Sniffer()
            sample = f.read(2048)
            f.seek(0)
            has_header = sniffer.has_header(sample)
            reader = csv.reader(f)
            rows = list(reader)
            if has_header and rows and rows[0] and "label_id" in rows[0][0].lower():
                # headered CSV
                out: Dict[int, str] = {}
                for i, row in enumerate(rows[1:], start=1):
                    if not row or len(row) < 2:
                        continue
                    try:
                        lid = int(row[0])
                    except Exception:
                        continue
                    name = row[1].strip()
                    out[lid] = name
                if out:
                    return out
            else:
                # maybe headered, but not the expected labels — fall through
                pass
    except Exception:
        # Fall back to raw reader
        pass

    # Raw MALPEM-style text (including versions where it's a .csv but one string column)
    with p.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # remove a possible banner line
    if lines and lines[0].lower().startswith("irtksegmenttable"):
        lines = lines[1:]

    lut: Dict[int, str] = {}
    for ln in lines:
        # expected: "<id> r g b 1 1 <name...>"
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            lid = int(parts[0])
        except Exception:
            continue
        # name is everything after the first 6 tokens if present, else after id
        if len(parts) >= 7:
            name = " ".join(parts[6:])
        else:
            name = " ".join(parts[1:])
        lut[lid] = name.strip()
    if not lut:
        raise ValueError("empty raw lut")

    return lut
