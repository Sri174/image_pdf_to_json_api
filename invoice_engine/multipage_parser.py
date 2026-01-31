"""
Production-grade multi-page invoice parser (offline, pdfplumber + pytesseract).

Provides: parse_multipage_invoice(pdf_path: str) -> dict

Design summary:
- Layer 1: layout-aware page stitching using spatial heuristics
- Layer 2: context-aware field extraction with constrained re-OCR for low confidence
- Layer 3: semantic reconstruction with business-rule validation and confidence waterfall

Constraints: pure Python, no cloud APIs.
"""
from typing import List, Dict, Any, Tuple
try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except Exception:
    pdfplumber = None
    _PDFPLUMBER_AVAILABLE = False
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import io
import re
import copy
import math
import statistics
import numpy as np

# Small helpers
def _num_or_none(s: str):
    try:
        return float(re.sub(r"[^0-9.\-]", "", s))
    except Exception:
        return None

def _normalize_text(t: str):
    if t is None:
        return None
    return re.sub(r"\s+", " ", t).strip()

def _pil_from_pdfpage(page, resolution=200):
    # pdfplumber Page.to_image -> PIL via .original
    try:
        im = page.to_image(resolution=resolution).original
        return im.convert("RGB")
    except Exception:
        # fallback: render crop to PNG bytes then open
        buf = page.crop((0, 0, page.width, page.height)).to_image(resolution=resolution).original
        return buf.convert("RGB")

def _ocr_cell_image(img: Image.Image, config: str = "", whitelist: str = None) -> Tuple[str, float]:
    # returns (text, confidence)
    try:
        if whitelist:
            config = (config + " -c tessedit_char_whitelist=" + whitelist).strip()
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
        texts = [t for t in data.get('text', []) if t and t.strip()]
        confs = [float(c) for c in data.get('conf', []) if c and c != '-1']
        text = " ".join(texts).strip() or None
        conf = statistics.mean(confs) if confs else 0.0
        return text, conf
    except Exception:
        try:
            text = pytesseract.image_to_string(img, config=config)
            return _normalize_text(text), 0.0
        except Exception:
            return None, 0.0

def _binarize_for_digits(img: Image.Image) -> Image.Image:
    # Try simple adaptive-like transform using PIL
    gray = ImageOps.grayscale(img)
    arr = np.array(gray).astype(np.uint8)
    # local mean threshold (very small Sauvola-like approximation)
    window = 25
    pad = window // 2
    padded = np.pad(arr, pad, mode='reflect')
    out = np.zeros_like(arr)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            win = padded[y:y+window, x:x+window]
            thresh = win.mean() * 0.85
            out[y, x] = 255 if arr[y, x] > thresh else 0
    return Image.fromarray(out)


def _detect_numeric_columns(stitched_rows: List[Dict[str, Any]], bin_width: int = 20) -> List[float]:
    """Detect candidate numeric column x positions by collecting x0 of numeric tokens and binning.
    Returns sorted list of x positions (left edges) for columns (likely qty, unit_price, amount from left->right).
    """
    positions = []
    num_re = re.compile(r'[-+]?\d{1,3}(?:[.,]\d{3})*(?:\.\d{1,2})?')
    for r in stitched_rows:
        for c in r.get('cells', []):
            t = c.get('text') or ''
            if num_re.search(t):
                positions.append(int(c.get('x0', 0)))
    if not positions:
        return []
    # bin positions
    minp, maxp = min(positions), max(positions)
    bins = {}
    for p in positions:
        key = int((p - minp) / bin_width)
        bins.setdefault(key, []).append(p)
    centers = [int(sum(v)/len(v)) for v in bins.values()]
    centers.sort()
    return centers


def _correct_product_name(name: str) -> str:
    """Apply lightweight OCR-noise corrections to product names.
    - remove non-printable characters
    - fix common OCR confusions
    - collapse multiple spaces and weird punctuation
    - title-case while preserving acronyms
    """
    if not name:
        return name
    s = name
    # replace common OCR mistakes
    subs = [
        (r'[\u2018\u2019`´]', "'"),
        (r'[\u201c\u201d\"]', '"'),
        (r'\|', 'I'),
        (r'0(?=[A-Za-z])', 'O'),
        (r'(?<=[A-Za-z])0', 'O'),
        (r'1(?=[A-Za-z])', 'I'),
        (r'5(?=[A-Za-z])', 'S'),
        (r'[^\x00-\x7F]+', ''),
    ]
    for p, rpl in subs:
        s = re.sub(p, rpl, s)
    # remove stray punctuation inside words
    s = re.sub(r'[^A-Za-z0-9\s\-\&\/]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    # fix casing: if all caps, keep caps; else title case but preserve words with 2+ caps
    if s.isupper():
        return s
    parts = []
    for w in s.split(' '):
        if re.search(r'[A-Z]{2,}', w):
            parts.append(w)
        else:
            parts.append(w.capitalize())
    return ' '.join(parts)



class MultiPageParser:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # tolerance px
        self.bottom_margin_threshold = self.config.get('bottom_margin_threshold_px', 20)
        self.horizontal_align_tol = self.config.get('horizontal_align_tol_px', 10)
        self.reocr_confidence_threshold = self.config.get('reocr_confidence_threshold', 70.0)

    # Layer 1: extract words and table-like rows with coordinates
    def layer1_extract(self, pdf_path: str) -> List[Dict[str, Any]]:
        if not _PDFPLUMBER_AVAILABLE:
            raise RuntimeError("pdfplumber is not installed. Install it with 'pip install pdfplumber' or update requirements.txt and run 'pip install -r requirements.txt'.")
        pages = []
        with pdfplumber.open(pdf_path) as doc:
            for pnum, page in enumerate(doc.pages, start=1):
                words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
                # normalize coords and group into rows by y midpoint clustering
                rows = {}
                for w in words:
                    midy = (float(w['top']) + float(w['bottom'])) / 2.0
                    # bucket by 6px
                    key = int(round(midy / 6.0))
                    rows.setdefault(key, []).append({
                        'text': w.get('text'),
                        'x0': float(w.get('x0', 0)),
                        'x1': float(w.get('x1', 0)),
                        'top': float(w.get('top', 0)),
                        'bottom': float(w.get('bottom', 0)),
                        'page': pnum
                    })
                # convert to ordered rows
                row_list = []
                for key in sorted(rows.keys()):
                    items = sorted(rows[key], key=lambda x: x['x0'])
                    row_text = " ".join([i['text'] for i in items if i.get('text')])
                    row_bbox = {
                        'x0': min(i['x0'] for i in items),
                        'x1': max(i['x1'] for i in items),
                        'top': min(i['top'] for i in items),
                        'bottom': max(i['bottom'] for i in items)
                    }
                    row_list.append({'text': row_text, 'cells': items, 'bbox': row_bbox, 'page': pnum})
                pages.append({'page_number': pnum, 'width': page.width, 'height': page.height, 'rows': row_list, 'page_obj': page})
        return pages

    # Stitch rows across page boundary when spatial heuristics indicate continuation
    def layer1_stitch(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        stitched_rows = []
        for i, pg in enumerate(pages):
            rows = pg['rows']
            for r in rows:
                r_copy = copy.deepcopy(r)
                r_copy['_page_span'] = (pg['page_number'], pg['page_number'])
                stitched_rows.append(r_copy)
            # check next page for stitching last row
            if i + 1 < len(pages) and rows:
                last = rows[-1]
                # bottom distance to page bottom
                dist_bottom = pages[i]['height'] - last['bbox']['bottom']
                if dist_bottom <= self.bottom_margin_threshold:
                    next_rows = pages[i+1]['rows']
                    if next_rows:
                        first_next = next_rows[0]
                        # check horizontal alignment of first content
                        if abs(first_next['bbox']['x0'] - last['bbox']['x0']) <= self.horizontal_align_tol:
                            # merge by concatenating text and updating bbox/page_span
                            merged = copy.deepcopy(last)
                            merged['text'] = (last['text'] + " " + first_next['text']).strip()
                            merged['bbox']['bottom'] = first_next['bbox']['bottom']
                            merged['_page_span'] = (last['page'], first_next['page'])
                            # replace last stitched with merged and mark to skip first_next
                            stitched_rows[-1] = merged
                            # mark to skip first_next when iterating pages (we don't remove actual next_rows here,
                            # layer2 will handle mapping by page_span)
        return stitched_rows

    # Layer 2: contextual field extraction
    def layer2_extract_fields(self, stitched_rows: List[Dict[str, Any]], pages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # Detect numeric column x-centers from layout to improve column assignment
        column_centers = _detect_numeric_columns(stitched_rows)
        # naive field candidates: try to identify qty, price, amount columns by numeric tokens on right
        items = []
        confidence_map = {}
        for r in stitched_rows:
            text = r.get('text') or ''
            # find numeric tokens assigned to cells and map to detected columns when possible
            nums = []
            cell_nums = []  # list of (x0, token, cleaned)
            num_re = re.compile(r'[-+]?\d{1,3}(?:[.,]\d{3})*(?:\.\d{1,2})?')
            for c in r.get('cells', []):
                t = c.get('text') or ''
                m = num_re.search(t)
                if m:
                    tok = m.group(0)
                    cell_nums.append((int(c.get('x0', 0)), tok))
                    nums.append(tok)

            qty = None
            unit_price = None
            amount = None
            if column_centers and cell_nums:
                # assign each numeric token to nearest column center
                assigned = {}
                for x0, tok in cell_nums:
                    # find nearest center
                    nearest = min(column_centers, key=lambda c: abs(c - x0))
                    assigned.setdefault(nearest, []).append(tok)
                # sort centers
                centers_sorted = sorted(assigned.keys())
                # assume rightmost center -> amount, next-left -> unit_price, next-left -> qty
                if centers_sorted:
                    vals = []
                    for c in centers_sorted:
                        # take last numeric token in that column
                        toks = assigned.get(c, [])
                        vals.append(_num_or_none(toks[-1].replace(',', '').replace(' ', '')) if toks else None)
                    # map from right
                    if len(vals) >= 1:
                        amount = vals[-1]
                    if len(vals) >= 2:
                        unit_price = vals[-2]
                    if len(vals) >= 3:
                        qty = vals[-3]
            else:
                # fallback to original rightmost-token heuristic
                if nums:
                    toks = [n.replace(',', '').replace(' ', '') for n in nums]
                    toks = [t for t in toks if re.search(r'\d', t)]
                    if toks:
                        if len(toks) >= 1:
                            amount = _num_or_none(toks[-1])
                        if len(toks) >= 2:
                            unit_price = _num_or_none(toks[-2])
                        if len(toks) >= 3:
                            qty = _num_or_none(toks[-3])
            # product name = text minus numeric tokens from right
            prod_text = re.sub(r'(' + r'|'.join([re.escape(n) for n in nums]) + r')\s*$', '', text).strip() if nums else text
            # Apply lightweight correction to product name
            prod_text = _correct_product_name(prod_text)

            # estimate confidence from presence of numeric tokens and lengths
            pre_conf = 80.0
            if len(text) > 200:
                pre_conf -= 10
            if re.search(r'[A-Za-z]{2,}', prod_text) is None:
                pre_conf -= 20

            item = {
                'product_name': _normalize_text(prod_text) or None,
                'qty': qty,
                'unit_price': unit_price,
                'amount': amount,
                '_page_span': r.get('_page_span', (r.get('page'), r.get('page'))),
                '_bbox': r.get('bbox')
            }
            items.append(item)
            # confidence map initial
            key = f"row_{len(items)-1}"
            confidence_map[key] = {
                'product_name_pre': pre_conf,
                'qty_pre': 80.0 if qty is not None else 30.0,
                'unit_price_pre': 80.0 if unit_price is not None else 30.0,
                'amount_pre': 90.0 if amount is not None else 10.0
            }

        # Re-OCR low confidence numeric cells with digit whitelist
        for idx, it in enumerate(items):
            key = f"row_{idx}"
            for fld in ('qty', 'unit_price', 'amount'):
                conf_pre = confidence_map[key].get(f'{fld}_pre', 0)
                if conf_pre < self.reocr_confidence_threshold:
                    # crop based on bbox and page
                    page_num = it['_page_span'][0]
                    # find page object
                    page = next((p for p in pages if p['page_number'] == page_num), None)
                    if not page:
                        continue
                    # crop region
                    bbox = it.get('_bbox')
                    if not bbox:
                        continue
                    try:
                        page_obj = page['page_obj']
                        im = _pil_from_pdfpage(page_obj)
                        # convert bbox to pixels; pdfplumber image uses same scale when rendering
                        left, top, right, bottom = int(bbox['x0']), int(bbox['top']), int(bbox['x1']), int(bbox['bottom'])
                        crop = im.crop((left, top, right, bottom)).resize((max(80, right-left), max(24, bottom-top)))
                        crop_proc = _binarize_for_digits(crop)
                        txt, conf = _ocr_cell_image(crop_proc, config='--psm 7', whitelist='0123456789.,')
                        val = _num_or_none(txt) if txt else None
                        if val is not None:
                            it[fld] = val
                            confidence_map[key][f'{fld}_post'] = conf
                        else:
                            confidence_map[key][f'{fld}_post'] = conf
                    except Exception:
                        continue

        return items, confidence_map

    # Layer 3: semantic reconstruction and validation
    def layer3_reconstruct(self, items: List[Dict[str, Any]], confidence_map: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        reconstructed = []
        report = {'reconstructed_fields': [], 'math_issues': []}
        for idx, it in enumerate(items):
            it2 = copy.deepcopy(it)
            key = f"row_{idx}"
            # If amount missing but qty and unit_price present, recalc
            if (it2.get('amount') is None or it2.get('amount') == 0) and it2.get('qty') is not None and it2.get('unit_price') is not None:
                calc = round(it2['qty'] * it2['unit_price'], 2)
                # allow tolerance ±5%
                it2['amount'] = calc
                it2['_reconstructed'] = True
                # confidence score: geometric mean of supporting fields
                qconf = confidence_map.get(key, {}).get('qty_post', confidence_map.get(key, {}).get('qty_pre', 50))
                pconf = confidence_map.get(key, {}).get('unit_price_post', confidence_map.get(key, {}).get('unit_price_pre', 50))
                score = float(max(5.0, min(99.0, math.sqrt(max(0.1, qconf/100.0) * max(0.1, pconf/100.0)) * 100)))
                it2['_confidence_score'] = round(score, 2)
                report['reconstructed_fields'].append({'row': idx, 'field': 'amount', 'reason': 'qty*unit_price', 'confidence': it2['_confidence_score']})
            else:
                it2['_reconstructed'] = False
                it2['_confidence_score'] = confidence_map.get(key, {}).get('amount_post', confidence_map.get(key, {}).get('amount_pre', 50))
            reconstructed.append(it2)

        # Business rule: ensure sum(line_items.amount) equals declared subtotal within tolerance
        total_sum = round(sum([(it.get('amount') or 0.0) for it in reconstructed]), 2)
        # declared subtotal may be present in a summary row; for now try to find row labelled subtotal
        declared = None
        for it in reconstructed:
            if it.get('product_name') and re.search(r'subtotal|total\s+amount|total\s+payable', it.get('product_name'), re.I):
                declared = it.get('amount')
                break
        if declared is not None and abs(total_sum - declared) > 0.01:
            report['math_issues'].append({'declared': declared, 'computed': total_sum, 'delta': round(total_sum-declared,2)})
        # else if declared missing, it's okay; caller may have separate summary

        return reconstructed, report

    def auto_crop_and_extract(self, pdf_path: str, max_crops: int = 12) -> Dict[str, Any]:
        """Try multiple preset crops on each page and return the best extraction result.

        Scoring heuristic: prefer results with more extracted line items (amount not None)
        and higher mean confidence when available.
        """
        if not _PDFPLUMBER_AVAILABLE:
            raise RuntimeError("pdfplumber is required for auto-cropping. Install pdfplumber.")

        best_result = None
        best_score = -1.0

        with pdfplumber.open(pdf_path) as doc:
            for pnum, page in enumerate(doc.pages, start=1):
                w, h = page.width, page.height
                # preset crops (x0,y0,x1,y1)
                presets = [
                    (0, 0, w, h),
                    (0, 0, w, h // 3),
                    (0, (h * 2) // 3, w, h),
                    (w // 4, h // 4, (w * 3) // 4, (h * 3) // 4),
                    (0, 0, w // 2, h),
                    (w // 2, 0, w, h),
                ]
                # 2x2 grid
                for yi in range(2):
                    for xi in range(2):
                        presets.append((int(xi * w / 2), int(yi * h / 2), int((xi + 1) * w / 2), int((yi + 1) * h / 2)))

                seen = set()
                crops = []
                for bbox in presets:
                    key = tuple(map(int, bbox))
                    if key in seen:
                        continue
                    seen.add(key)
                    crops.append(key)
                    if len(crops) >= max_crops:
                        break

                for bbox in crops:
                    try:
                        cropped_page = page.crop(bbox)
                        words = cropped_page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
                        # assemble rows from words
                        rows = {}
                        for wobj in words:
                            midy = (float(wobj['top']) + float(wobj['bottom'])) / 2.0
                            key = int(round(midy / 6.0))
                            rows.setdefault(key, []).append({
                                'text': wobj.get('text'),
                                'x0': float(wobj.get('x0', 0)),
                                'x1': float(wobj.get('x1', 0)),
                                'top': float(wobj.get('top', 0)),
                                'bottom': float(wobj.get('bottom', 0)),
                                'page': pnum
                            })
                        row_list = []
                        for k in sorted(rows.keys()):
                            items = sorted(rows[k], key=lambda x: x['x0'])
                            row_text = " ".join([i['text'] for i in items if i.get('text')])
                            row_bbox = {
                                'x0': min(i['x0'] for i in items),
                                'x1': max(i['x1'] for i in items),
                                'top': min(i['top'] for i in items),
                                'bottom': max(i['bottom'] for i in items)
                            }
                            row_list.append({'text': row_text, 'cells': items, 'bbox': row_bbox, 'page': pnum})

                        pages_crop = [{'page_number': pnum, 'width': bbox[2] - bbox[0], 'height': bbox[3] - bbox[1], 'rows': row_list, 'page_obj': cropped_page}]
                        stitched = self.layer1_stitch(pages_crop)
                        items, conf_map = self.layer2_extract_fields(stitched, pages_crop)
                        reconstructed, report = self.layer3_reconstruct(items, conf_map)

                        # build result
                        result = {'line_items': [], 'summary': {}, '_debug': {'confidence_map': conf_map, 'stitch_count': len(stitched), 'page_count': 1, 'reconstruction_report': report, 'crop_bbox': bbox, 'page': pnum}}
                        for it in reconstructed:
                            entry = {
                                'product_name': it.get('product_name'),
                                'quantity': it.get('qty'),
                                'unit_price': it.get('unit_price'),
                                'amount': it.get('amount'),
                                '_reconstructed': it.get('_reconstructed', False),
                                '_confidence_score': it.get('_confidence_score')
                            }
                            result['line_items'].append(entry)
                        result['summary']['total_amount'] = round(sum([li.get('amount') or 0.0 for li in result['line_items']]), 2)

                        # scoring: prefer more amounts found, then higher mean confidence
                        amounts_count = sum(1 for li in result['line_items'] if li.get('amount'))
                        mean_conf = 0.0
                        conf_vals = [float(it.get('_confidence_score') or 0.0) for it in reconstructed]
                        if conf_vals:
                            mean_conf = sum(conf_vals) / len(conf_vals)
                        score = amounts_count * 100.0 + mean_conf

                        if score > best_score:
                            best_score = score
                            best_result = result
                    except Exception:
                        continue

        if best_result:
            best_result['_debug']['auto_crop_score'] = best_score
            return best_result
        return {'line_items': [], 'summary': {'total_amount': 0}, '_debug': {'note': 'auto_crop_no_result'}}

    def parse(self, pdf_path: str) -> Dict[str, Any]:
        pages = self.layer1_extract(pdf_path)
        stitched = self.layer1_stitch(pages)
        items, conf_map = self.layer2_extract_fields(stitched, pages)
        reconstructed, report = self.layer3_reconstruct(items, conf_map)

        # Build final schema-compliant result
        result = {
            'line_items': [],
            'summary': {},
            '_debug': {
                'confidence_map': conf_map,
                'stitch_count': len(stitched),
                'page_count': len(pages),
                'reconstruction_report': report
            }
        }
        for it in reconstructed:
            entry = {
                'product_name': it.get('product_name'),
                'quantity': it.get('qty'),
                'unit_price': it.get('unit_price'),
                'amount': it.get('amount'),
                '_reconstructed': it.get('_reconstructed', False),
                '_confidence_score': it.get('_confidence_score')
            }
            result['line_items'].append(entry)

        # populate summary total as computed and include validation
        result['summary']['total_amount'] = round(sum([li.get('amount') or 0.0 for li in result['line_items']]), 2)
        result['_debug']['validation'] = report

        return result


def parse_multipage_invoice(pdf_path: str) -> Dict[str, Any]:
    parser = MultiPageParser()
    res = parser.parse(pdf_path)
    # If no line items found, try automated crop-presets to salvage extraction
    try:
        # only run auto-crop for multi-page documents (avoid cropping single-page images)
        page_count = res.get('_debug', {}).get('page_count', 1)
        if (not res.get('line_items')) and _PDFPLUMBER_AVAILABLE and page_count > 1:
            try:
                auto = parser.auto_crop_and_extract(pdf_path)
                # choose the best result (auto_crop returns full result)
                if auto and auto.get('line_items'):
                    return auto
            except Exception:
                pass
    except Exception:
        pass
    return res

