"""
structured_extractor.py
----------------------
A higher-accuracy, structured PDF/image -> JSON extractor.

Strategy:
1) Try `pdfplumber` to extract native text and tables (best for born-digital PDFs).
2) If pdfplumber finds tables, map table columns to the invoice schema using header fuzzy-matching.
3) Fallback: detect table regions with OpenCV line detection on rasterized pages and OCR each cell.

This module uses `invoice_engine.local_extraction._load_schema_template` to
produce schema-compatible JSON and `invoice_engine.local_extraction._ocr_with_variants`
for OCR (fast mode used by default).

Note: add `pdfplumber` to your environment if not present: `pip install pdfplumber`.
"""
from typing import List, Dict, Any, Optional
import io
import re
from pathlib import Path

from PIL import Image
import pytesseract
import cv2
import numpy as np

try:
    import pdfplumber
except Exception:
    pdfplumber = None

from invoice_engine.local_extraction import _load_schema_template, _ocr_with_variants


def _colname_score(colname: str, target: str) -> int:
    """Simple fuzzy score for header matching."""
    if not colname:
        return 0
    s = colname.lower()
    t = target.lower()
    # exact contains
    if t in s or s in t:
        return 10
    # word overlap
    sset = set(re.findall(r"\w+", s))
    tset = set(re.findall(r"\w+", t))
    return len(sset & tset)


def _map_table_row_to_item(headers: List[str], row: List[str]) -> Dict[str, Any]:
    """Map a table row to a line item by header heuristics."""
    item = {
        "line_number": None,
        "prod_code": None,
        "barcode": None,
        "product_name": None,
        "description": None,
        "packing": None,
        "unit": None,
        "unit_of_measure": None,
        "qty": None,
        "quantity": None,
        "unit_price": None,
        "gross_amount": None,
        "discount": None,
        "taxed": False,
        "vat_percent": None,
        "net_value": None,
        "excise": None,
        "total_incl_excise": None,
        "vat_amount": None,
        "amount": None,
    }
    for h, cell in zip(headers, row):
        if not cell:
            continue
        # candidate numeric
        num = None
        m = re.search(r"[-+]?[0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]{1,2})?", cell)
        if m:
            try:
                num = float(m.group(0).replace(',', '').replace(' ', ''))
            except Exception:
                num = None
        hlow = (h or "").lower()
        if any(k in hlow for k in ("qty", "quantity", "qty.")) and num is not None:
            item['qty'] = int(num) if float(num).is_integer() else num
            item['quantity'] = item['qty']
        elif any(k in hlow for k in ("price", "unit price", "rate", "unit_price")) and num is not None:
            item['unit_price'] = num
        elif any(k in hlow for k in ("amount", "total", "line total", "value")) and num is not None:
            item['amount'] = num
        elif any(k in hlow for k in ("code", "sku", "prod", "product code", "barcode")):
            digits = re.sub(r"\D", "", cell)
            if len(digits) >= 6:
                item['prod_code'] = digits
                item['barcode'] = digits
            else:
                item['product_name'] = cell.strip()
                item['description'] = cell.strip()
        else:
            # fall back: treat as description if nothing else matched
            if not item['description']:
                item['description'] = cell.strip()
                item['product_name'] = cell.strip()
    return item


def _table_to_schema(table: List[List[str]]) -> Dict[str, Any]:
    template = _load_schema_template()
    if not table or not any(table):
        return template
    # assume first non-empty row is header
    header = table[0]
    rows = table[1:]
    items = []
    for r in rows:
        if not any(c and c.strip() for c in r):
            continue
        item = _map_table_row_to_item(header, r)
        items.append(item)
    if items:
        template['line_items'] = items
        try:
            computed_total = sum([i.get('amount') or 0.0 for i in items])
        except Exception:
            computed_total = None
        template.setdefault('summary', {})
        template['summary']['total_amount'] = round(computed_total, 2) if computed_total is not None else None
    return template


def parse_pdf_to_json(pdf_path: str, fast: bool = True) -> Dict[str, Any]:
    """Primary entrypoint: try pdfplumber tables first, else fallback to image table detection.

    Returns a schema-matching JSON dict.
    """
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(pdf_path)

    # Try pdfplumber first (best for digital PDFs)
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(p)) as doc:
                # try to find first useful table across pages
                for page in doc.pages:
                    try:
                        tables = page.extract_tables()
                    except Exception:
                        tables = []
                    if tables:
                        # pick the largest table
                        tables_sorted = sorted(tables, key=lambda t: len(t))
                        best = tables_sorted[-1]
                        return _table_to_schema(best)
        except Exception:
            pass

    # Fallback: rasterize pages and detect table by lines
    try:
        import pdf2image
    except Exception:
        pdf2image = None
    if pdf2image is None:
        # last resort: return empty template
        return _load_schema_template()

    pages = pdf2image.convert_from_path(str(p), dpi=200)
    for pil_page in pages:
        tbl = _detect_table_and_ocr(pil_page, fast=fast)
        if tbl and tbl.get('line_items'):
            return tbl
    return _load_schema_template()


def _detect_table_and_ocr(pil_img: Image.Image, fast: bool = True) -> Dict[str, Any]:
    """Detect table grid with OpenCV and OCR each cell."""
    img = np.array(pil_img.convert('L'))
    # binarize
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert for line detection (lines are darker)
    inv = 255 - th
    # detect horizontal and vertical lines
    horiz = inv.copy()
    vert = inv.copy()
    cols = horiz.shape[1]
    horiz_size = max(10, cols // 40)
    horiz_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
    horiz = cv2.erode(horiz, horiz_structure)
    horiz = cv2.dilate(horiz, horiz_structure)

    rows = vert.shape[0]
    vert_size = max(10, rows // 40)
    vert_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))
    vert = cv2.erode(vert, vert_structure)
    vert = cv2.dilate(vert, vert_structure)

    grid = cv2.addWeighted(horiz, 0.5, vert, 0.5, 0.0)
    # find contours of table areas
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}
    # pick largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    # crop table region
    crop = pil_img.crop((x, y, x + w, y + h))

    # attempt to split into rows by horizontal projection
    gray = cv2.cvtColor(np.array(crop.convert('RGB')), cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    projection = np.sum(bw == 0, axis=1)
    # find valleys to split rows
    thresh = max(1, int(np.max(projection) * 0.05))
    splits = []
    in_gap = False
    for i, v in enumerate(projection):
        if v <= thresh and not in_gap:
            # start of gap
            in_gap = True
            splits.append(i)
        elif v > thresh and in_gap:
            in_gap = False
    # crude row bounding boxes
    row_bounds = []
    if len(splits) >= 2:
        # make pairs
        for i in range(0, len(splits) - 1, 2):
            r0 = splits[i]
            r1 = splits[i + 1]
            row_bounds.append((r0, r1))
    else:
        # fallback: single row containing whole crop
        row_bounds = [(0, crop.size[1])]

    # OCR each row and try to split into columns using whitespace
    table_rows = []
    for (r0, r1) in row_bounds:
        rimg = crop.crop((0, r0, crop.size[0], r1))
        # quick OCR to get line text
        txt, dbg = _ocr_with_variants(rimg, None, fast)
        # split by multiple spaces or tab-like whitespace to columns
        cols = re.split(r"\s{2,}|\t|\|", txt.strip())
        table_rows.append([c.strip() for c in cols if c is not None])

    if table_rows:
        return _table_to_schema(table_rows)
    return {}
