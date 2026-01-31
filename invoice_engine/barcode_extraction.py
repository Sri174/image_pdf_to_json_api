"""
barcode_extraction.py
---------------------
Deterministic barcode / QR extraction using pyzbar + OpenCV.

Functions:
- extract_codes_from_bytes(image_bytes) -> list of {type, value, confidence}

Requirements: opencv-python, pyzbar, numpy
"""
from typing import List, Dict
import io
import numpy as np
import cv2
from pyzbar.pyzbar import decode, ZBarSymbol
import os
from contextlib import contextmanager


@contextmanager
def _suppress_stderr():
    """Temporarily redirect low-level stderr (fd 2) to os.devnull to silence C-level errors from zbar."""
    try:
        devnull = os.open(os.devnull, os.O_RDWR)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    finally:
        try:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        except Exception:
            pass


def extract_codes_from_bytes(image_bytes: bytes) -> List[Dict]:
    """Decode barcodes/QR codes from raw image bytes.

    Returns a list of dicts:
      {"type": "QR_CODE", "value": "...", "confidence": 1.0}

    Deterministic: confidence is always 1.0 when decoded.
    If no codes found, returns an empty list.
    """
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []
        # pyzbar.decode accepts numpy arrays — suppress C-level stderr spam from zbar
        with _suppress_stderr():
            decoded = decode(img)
        codes = []
        for d in decoded:
            code_type = d.type
            try:
                value = d.data.decode('utf-8')
            except Exception:
                value = d.data.decode('latin-1', errors='ignore')
            codes.append({
                "type": code_type,
                "value": value,
                "confidence": 1.0
            })
        return codes
    except Exception:
        return []


def extract_codes_from_images(image_bytes_list: List[bytes]) -> List[Dict]:
    """Decode barcodes/QR codes from a list of image bytes (one per page).

    Returns a deduplicated list of dicts with page numbers:
      {"type": "QR_CODE", "value": "...", "page": 1, "confidence": 1.0}

    Deduplication is performed on (type, value) and the first page occurrence is kept.
    """
    supported = {"QR_CODE", "CODE128", "EAN", "PDF417", "DATA_MATRIX"}
    def normalize_type(t: str) -> str:
        tu = (t or "").upper()
        if tu in ("QRCODE", "QR_CODE", "QR"):
            return "QR_CODE"
        if "CODE128" in tu:
            return "CODE128"
        if tu.startswith("EAN"):
            return "EAN"
        if "PDF417" in tu or "PDF_417" in tu:
            return "PDF417"
        if "DATA" in tu and "MATRIX" in tu:
            return "DATA_MATRIX"
        return tu

    seen = set()
    out = []
    for idx, image_bytes in enumerate(image_bytes_list, start=1):
        try:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            with _suppress_stderr():
                decoded = decode(img)
            for d in decoded:
                raw_type = d.type
                ctype = normalize_type(raw_type)
                if ctype not in supported:
                    # skip unsupported barcode families
                    continue
                try:
                    value = d.data.decode('utf-8')
                except Exception:
                    value = d.data.decode('latin-1', errors='ignore')
                key = (ctype, value)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "type": ctype,
                    "value": value,
                    "page": idx,
                    "confidence": 1.0
                })
        except Exception:
            continue
    return out
