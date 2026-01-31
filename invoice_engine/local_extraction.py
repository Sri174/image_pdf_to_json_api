"""
local_extraction.py
-------------------
Simple local OCR + heuristics extractor.

Uses `pytesseract` to extract text and regex rules to find
invoice number, invoice date, and total amount. Returns a
JSON object matching `universal_schema.json` structure (keys present).

This is a best-effort fallback to reduce Vision LLM calls.
"""
from typing import Dict, Any, Optional
import json
import re
from io import BytesIO
from PIL import Image, ImageFilter
import pytesseract
import cv2
import numpy as np
from pytesseract import Output


def _ocr_with_variants(pil_img: Image.Image, try_lang: Optional[str] = None, fast: bool = True):
    """Run preprocessing + tesseract-config variants and return best (text, debug dict).

    When `fast` is True this uses a small set of high-quality, fast variants
    (fewer preprocessing passes and a single reliable PSM/OEM) to dramatically
    reduce runtime while keeping good accuracy. Set `fast=False` to run the
    full ensemble (slower).
    """
    import cv2
    from pytesseract import Output as _Out
    w, h = pil_img.size
    # ensure RGB for consistent cv2 operations
    try:
        pil_img = pil_img.convert('RGB')
    except Exception:
        pass

    # base resize scale to improve OCR if image is small
    base_scale = 1.0
    if max(w, h) < 1400:
        base_scale = 1600.0 / max(w, h)

    # define preprocessing funcs
    def prep_none(img):
        return img

    def prep_clahe(img):
        arr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        out = clahe.apply(arr)
        return Image.fromarray(out)

    def prep_bilateral(img):
        arr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
        out = cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        return Image.fromarray(out)

    def prep_unsharp(img):
        return img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

    def prep_median(img):
        arr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2GRAY)
        out = cv2.medianBlur(arr, 3)
        return Image.fromarray(out)

    def prep_morph(img):
        arr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2GRAY)
        kernel = np.ones((3,3), np.uint8)
        out = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
        return Image.fromarray(out)

    def prep_adapt_small(img):
        arr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2GRAY)
        out = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9)
        return Image.fromarray(out)

    # Choose a small, fast, high-quality set when requested
    if fast:
        preps = [prep_none, prep_clahe, prep_unsharp]
        # Prefer a single stable PSM for mixed text/layout (6 = assume a block of text)
        psm_opts = [6]
        oem_opts = [1]
    else:
        preps = [prep_none, prep_clahe, prep_bilateral, prep_unsharp, prep_median, prep_morph, prep_adapt_small]
        # tesseract configs to try (include more PSMs for layout variants)
        psm_opts = [3, 4, 6, 7, 11]
        oem_opts = [1, 3]

    best = None
    best_score = -1.0
    best_debug = {}

    for prep in preps:
        try:
            img_p = pil_img.copy()
            # apply base scale
            if base_scale != 1.0:
                new_w, new_h = int(w * base_scale), int(h * base_scale)
                img_p = img_p.resize((new_w, new_h), Image.LANCZOS)
            img_proc = prep(img_p)
            # ensure RGB or L for tesseract
            if img_proc.mode not in ('L', 'RGB'):
                img_proc = img_proc.convert('RGB')
        except Exception:
            continue

        for psm in psm_opts:
            for oem in oem_opts:
                cfg = f"--oem {oem} --psm {psm}"
                try:
                    data = pytesseract.image_to_data(img_proc, output_type=_Out.DICT, config=cfg, lang=try_lang) if try_lang else pytesseract.image_to_data(img_proc, output_type=_Out.DICT, config=cfg)
                except Exception:
                    try:
                        data = pytesseract.image_to_data(img_proc, output_type=_Out.DICT, config=cfg)
                    except Exception:
                        continue
                texts = [t for t in data.get('text', []) if t and t.strip()]
                confs = []
                for c in data.get('conf', []):
                    try:
                        conff = float(c)
                        if conff >= 0:
                            confs.append(conff)
                    except Exception:
                        continue
                text = "\n".join([" ".join([texts[i] for i in range(j, min(j+20, len(texts)))]) for j in range(0, len(texts), 20)]) if texts else ""
                mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
                # numeric token presence
                nums = re.findall(r"[-+]?\d{1,3}(?:[.,]\d{3})*(?:\.\d{1,2})?", text or "")
                score = mean_conf + len(nums) * 15.0
                if score > best_score:
                    best_score = score
                    best = text or ""
                    best_debug = {
                        'config': cfg,
                        'prep': prep.__name__,
                        'mean_conf': mean_conf,
                        'numeric_tokens': len(nums),
                        'n_words': len(texts)
                    }
    # fallback: try a simple tesseract run if nothing found
    if best is None:
        try:
            data = pytesseract.image_to_data(pil_img, output_type=_Out.DICT)
            texts = [t for t in data.get('text', []) if t and t.strip()]
            confs = []
            for c in data.get('conf', []):
                try:
                    conff = float(c)
                    if conff >= 0:
                        confs.append(conff)
                except Exception:
                    continue
            text = "\n".join([" ".join([texts[i] for i in range(j, min(j+20, len(texts)))]) for j in range(0, len(texts), 20)]) if texts else ""
            mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
            nums = re.findall(r"[-+]?\d{1,3}(?:[.,]\d{3})*(?:\.\d{1,2})?", text or "")
            best = text
            best_debug = {'config': 'fallback', 'prep': 'none', 'mean_conf': mean_conf, 'numeric_tokens': len(nums), 'n_words': len(texts)}
        except Exception:
            best = ""
            best_debug = {}

    return best or "", best_debug


def _normalize_ocr_text(text: str) -> str:
    """Apply simple OCR normalization rules to reduce common character errors.

    This is intentionally small and conservative to avoid changing numeric tokens.
    """
    if not text:
        return text
    # Common substitutions observed in invoices
    rules = [
        (r"\bY0G\b", "YOG"),
        (r"\bHILK\b", "MILK"),
        (r"\bWILK\b", "MILK"),
        (r"\bYOR\b", "YOG"),
    ]
    # Replace currency-like garbage characters often introduced by OCR
    text = text.replace('¥', 'Y')
    # Apply word-preserving regex replacements
    for pat, sub in rules:
        try:
            text = re.sub(pat, sub, text, flags=re.IGNORECASE)
        except Exception:
            pass
    # Normalize whitespace
    text = re.sub(r"[ \t\x0b\f]+", " ", text)
    return text


def _load_schema_template() -> Dict[str, Any]:
    with open("invoice_engine/universal_schema.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _find_invoice_number(text: str) -> Optional[str]:
    # Broader set of labels across languages and formats
    patterns = [
        r"(?im)invoice\s*(?:number|no|#)[:\s]*([A-Z0-9\-\/_\.]+)",
        r"(?im)inv(?:ice)?\s*[:#]\s*([A-Z0-9\-\/_\.]+)",
        r"(?im)factura\s*(?:n(?:u?mero)?|no|#)[:\s]*([A-Z0-9\-\/_\.]+)",
        r"(?im)rechnung\s*(?:nr|nr\.|no|#)[:\s]*([A-Z0-9\-\/_\.]+)",
        r"(?im)folio[:\s]*([A-Z0-9\-\/_\.]+)",
        r"(?im)no\.\s*([A-z0-9\-\/_\.]+)\b"
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).strip()
    # As a last resort, find lines that look like 'INV12345' or '12345' near 'Invoice'
    m = re.search(r"(?im)invoice[:\s]*([A-Z0-9\-\/_\.]{4,})", text)
    if m:
        return m.group(1).strip()
    return None


def _find_date(text: str) -> Optional[str]:
    # Common numeric dates like 01/02/2023 or 2023-12-31
    m = re.search(r"(\d{4}[\-]\d{1,2}[\-]\d{1,2})", text)
    if m:
        return m.group(1)
    m = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", text)
    if m:
        return m.group(1)

    # Dates with month names (supporting multiple language abbreviations)
    months = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch|z)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|ene|abr|ago|dic|mär|mai|okt|dez)"
    m = re.search(rf"(\d{{1,2}}\s+{months}\s+\d{{4}})", text, flags=re.I)
    if m:
        return m.group(1)

    # Look for key labels with nearby dates
    m = re.search(r"(?im)(?:date|fecha|datum)[:\s]*([\w\d\./\- ]{6,30})", text)
    if m:
        return m.group(1).strip()
    return None


def _find_total(text: str) -> Optional[float]:
    # Look for labeled totals across languages and formats
    patterns = [
        r"(?im)(?:total(?: amount)?|amount due|balance due|net payable|total a pagar|importe total|montant|betrag)[:\s]*\$?([\d,.]+)",
        r"(?im)(?:subtotal|sub-total|subtotal amount)[:\s]*\$?([\d,.]+)",
        r"(?im)(?:amount|importe|montant)[:\s]*\$?([\d,.]+)"
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            val = m.group(1).replace(".", "*").replace(",", "").replace("*", ".")
            try:
                return float(val)
            except Exception:
                continue

    # fallback: find currency-like numbers and assume the largest/last is the total
    m_all = re.findall(r"\$?([\d,.]+)\s*(?:€|USD|EUR|GBP|\b)??", text)
    if m_all:
        # pick the last numeric-looking token and coerce
        for candidate in reversed(m_all):
            val = candidate.replace(".", "*").replace(",", "").replace("*", ".")
            try:
                return float(val)
            except Exception:
                continue
    return None


def local_extract_invoice(image_bytes: bytes, lang: Optional[str] = None, fast: bool = True) -> Dict[str, Any]:
    """Run local OCR and return a best-effort invoice JSON.

    If no text can be read, returns the schema with mostly nulls.
    """
    template = _load_schema_template()
    raw_text = ""
    debug = {"words": [], "avg_confidence": None}
    try:
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        # Preprocess with OpenCV for better OCR: grayscale, resize, denoise, adaptive threshold
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = 1600.0 / max(h, w) if max(h, w) > 0 else 1.0
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # denoise
        gray = cv2.fastNlMeansDenoising(gray, None, h=10)
        # adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 15)
        proc_pil = Image.fromarray(thresh)

        # Try tesseract with provided lang, fall back if it errors
        try_langs = [lang] if lang else [None]
        tried = set()
        ocr_text = ""
        ocr_data = None
        for try_lang in try_langs:
            key = try_lang or "default"
            if key in tried:
                continue
            tried.add(key)
            try:
                # Use OCR helper (fast by default) to get best text/config
                best_text, best_dbg = _ocr_with_variants(proc_pil, try_lang, fast)
                ocr_text = best_text or ""
                debug['words'] = []
                debug['avg_confidence'] = best_dbg.get('mean_conf') if isinstance(best_dbg, dict) else None
                # store variant info
                debug['best_ocr_variant'] = best_dbg
                raw_text = ocr_text
                break
            except Exception:
                raw_text = ""
                ocr_data = None
                continue
        # Normalize OCR text conservatively to improve heuristics
        text = _normalize_ocr_text(raw_text)
        # Run a digit-only OCR pass to improve numeric token detection (totals, amounts)
        numeric_candidates = []
        try:
            # try to get a binarized/narrow OCR for digits
            try:
                num_txt = pytesseract.image_to_string(proc_pil, config='--psm 7 -c tessedit_char_whitelist=0123456789.,')
            except Exception:
                num_txt = pytesseract.image_to_string(proc_pil, config='--psm 7')
            if num_txt:
                numeric_candidates = re.findall(r"[-+]?\d{1,3}(?:[.,]\d{3})*(?:\.\d{1,2})?", num_txt)
        except Exception:
            numeric_candidates = []
        # If digit-only OCR found candidates, and main detected total is missing, use largest numeric as fallback
        if numeric_candidates and detected_total is None:
            try:
                nums = [float(n.replace(',', '').replace(' ', '')) for n in numeric_candidates]
                if nums:
                    # prefer the largest numeric token as total candidate
                    cand = max(nums)
                    total_amount = cand if total_amount is None else total_amount
            except Exception:
                pass
    except Exception:
        text = ""

    invoice_number = _find_invoice_number(text)
    invoice_date = _find_date(text)
    total_amount = _find_total(text)

    # Attempt to parse line items from OCR raw text heuristically
    def _parse_line_items(raw: str):
        """Heuristic parser that returns items conforming to the universal schema.

        Looks for product codes (12-13 digit UPC/EAN), extracts nearby numeric tokens
        and maps them to `quantity`, `unit_price`, and `amount` where possible.
        """
        items = []
        lines = [l for l in raw.splitlines() if l.strip()]
        code_re = re.compile(r"\b(\d{12,13})\b")
        # decimal number tokens like 189.68 or 1,602.54
        decimal_re = re.compile(r"[\d]{1,3}(?:[\d,]*\d)?(?:\.\d{1,2})")
        int_re = re.compile(r"\b\d{1,5}\b")

        for ln in lines:
            m = code_re.search(ln)
            if not m:
                # Try to find lines where code is at start separated by spaces (common in poor OCR)
                parts = ln.split()
                found_idx = None
                for i, p in enumerate(parts):
                    if re.fullmatch(r"\d{12,13}", re.sub(r"[^0-9]", "", p)):
                        found_idx = i
                        break
                if found_idx is not None:
                    # rebuild a normalized line where code is found
                    prod_code = re.sub(r"[^0-9]", "", parts[found_idx])
                else:
                    continue
            else:
                prod_code = m.group(1)

            # Find all decimal-like tokens in the line (rightmost usually total)
            decs = decimal_re.findall(ln)
            decs = [d.replace(',', '') for d in decs]
            amount = None
            unit_price = None
            if decs:
                try:
                    amount = float(decs[-1])
                except Exception:
                    amount = None
                if len(decs) >= 2:
                    try:
                        unit_price = float(decs[-2])
                    except Exception:
                        unit_price = None

            # Quantity heuristics: integer tokens near code or before unit_price
            qty = None
            # Try tokens between code and first decimal
            try:
                post = ln[ln.find(prod_code) + len(prod_code):]
                ints = int_re.findall(post)
                for t in ints:
                    ti = int(t)
                    if 0 < ti <= 100000:
                        qty = ti
                        break
            except Exception:
                qty = None

            # Build description: text between code and first numeric token
            desc = ln
            try:
                # cut off at first decimal token occurrence to avoid trailing totals
                first_dec_match = decimal_re.search(ln)
                if first_dec_match:
                    desc = ln[:first_dec_match.start()]
                # remove the code itself
                desc = desc.replace(prod_code, "")
                desc = re.sub(r"[^\w\s\-\,\.]", " ", desc).strip()
            except Exception:
                desc = None

            item = {
                "line_number": None,
                "prod_code": prod_code,
                "barcode": prod_code,
                "product_name": desc or None,
                "description": desc or None,
                "packing": None,
                "unit": None,
                "unit_of_measure": None,
                "qty": qty,
                "quantity": qty,
                "unit_price": unit_price,
                "gross_amount": None,
                "discount": None,
                "taxed": False,
                "vat_percent": None,
                "net_value": None,
                "excise": None,
                "total_incl_excise": None,
                "vat_amount": None,
                "amount": amount,
            }
            items.append(item)
        return items

    def _parse_line_items_positional(raw: str, words: list):
        """Use positional OCR word boxes to group rows and extract structured columns.

        Returns items in schema shape. Expects `words` as list of dicts with keys
        `text`, `left`, `top`, `width`, `height` produced earlier.
        """
        if not words:
            return []

        # bucket words into rows by y coordinate (top) with tolerance
        rows = {}
        for w in words:
            try:
                top = int(round(float(w.get('top', 0))))
            except Exception:
                top = 0
            # bucket size 10 px
            bucket = int(round(top / 10.0) * 10)
            rows.setdefault(bucket, []).append(w)

        items = []
        decimal_re = re.compile(r"\d{1,3}(?:[\d,]*\d)?(?:\.\d{1,2})")
        int_re = re.compile(r"\b\d{1,5}\b")

        for y in sorted(rows.keys()):
            row_words = sorted(rows[y], key=lambda x: int(x.get('left', 0)))
            row_text = " ".join([w['text'] for w in row_words if w.get('text')])

            # try find a product code in row words
            prod_code = None
            for w in row_words:
                digits = re.sub(r"\D", "", w.get('text', ''))
                if len(digits) in (12, 13):
                    prod_code = digits
                    code_left = int(w.get('left', 0))
                    break
            if not prod_code:
                # skip rows without product codes
                continue

            # collect numeric tokens with their x positions
            nums = []
            for w in row_words:
                txt = w.get('text', '')
                for m in decimal_re.findall(txt):
                    nums.append((int(w.get('left', 0)), m))
            nums = sorted(nums, key=lambda x: x[0])
            amounts = [n[1].replace(',', '') for n in nums]

            amount = None
            unit_price = None
            if amounts:
                try:
                    amount = float(amounts[-1])
                except Exception:
                    amount = None
                if len(amounts) >= 2:
                    try:
                        unit_price = float(amounts[-2])
                    except Exception:
                        unit_price = None

            # find qty as integer tokens between code position and first numeric token
            qty = None
            # determine x of first numeric token if exists
            first_num_x = nums[0][0] if nums else None
            for w in row_words:
                try:
                    lx = int(w.get('left', 0))
                except Exception:
                    lx = 0
                if first_num_x and lx >= code_left and lx < first_num_x:
                    if int_re.fullmatch(w.get('text', '')):
                        try:
                            qty = int(w.get('text'))
                            break
                        except Exception:
                            pass
            # fallback: any int in row
            if qty is None:
                for w in row_words:
                    if int_re.fullmatch(w.get('text', '')):
                        try:
                            qty = int(w.get('text'))
                            break
                        except Exception:
                            pass

            # description: words between code and first numeric token (by x)
            desc = None
            try:
                desc_parts = []
                for w in row_words:
                    lx = int(w.get('left', 0))
                    if lx > code_left and (first_num_x is None or lx < first_num_x):
                        desc_parts.append(w.get('text', ''))
                desc = " ".join(desc_parts).strip() or None
            except Exception:
                desc = None

            item = {
                "line_number": None,
                "prod_code": prod_code,
                "barcode": prod_code,
                "product_name": desc or None,
                "description": desc or None,
                "packing": None,
                "unit": None,
                "unit_of_measure": None,
                "qty": qty,
                "quantity": qty,
                "unit_price": unit_price,
                "gross_amount": None,
                "discount": None,
                "taxed": False,
                "vat_percent": None,
                "net_value": None,
                "excise": None,
                "total_incl_excise": None,
                "vat_amount": None,
                "amount": amount,
            }
            items.append(item)
        return items

    # Prefer positional parsing when word boxes are available
    parsed_items = []
    try:
        parsed_items = _parse_line_items_positional(text, debug.get("words", []))
    except Exception:
        parsed_items = []
    if not parsed_items:
        parsed_items = _parse_line_items(text)
    # If we found line items, populate template['line_items'] and compute a summed total
    if parsed_items:
        template["line_items"] = parsed_items
        try:
            computed_total = sum([i.get("amount") or 0.0 for i in parsed_items])
        except Exception:
            computed_total = None
    else:
        computed_total = None

    # Detect explicit total/subtotal lines in the OCR raw text (prefer these over computed totals)
    detected_total = None
    detected_total_line = None
    try:
        # Look for common total labels and last numeric token on that line.
        # Add more variants (amount due, balance due, total due, net amount, invoice total)
        total_patterns = [r"(?im)total(?:\s+payable|\s+amount|\s+incl|\s+inclusive|)[:\s]*([\d,]+\.?\d{0,2})",
                          r"(?im)grand\s+total[:\s]*([\d,]+\.?\d{0,2})",
                          r"(?im)total\s*:\s*([\d,]+\.?\d{0,2})",
                          r"(?im)subtotal[:\s]*([\d,]+\.?\d{0,2})",
                          r"(?im)total\s+payable[:\s]*([\d,]+\.?\d{0,2})",
                          r"(?im)delivery\s*/\s*total\s+payable[:\s]*([\d,]+\.?\d{0,2})",
                          r"(?im)amount\s+payable[:\s]*([\d,]+\.?\d{0,2})",
                          r"(?im)amount\s+due[:\s]*([\d,]+\.?\d{0,2})",
                          r"(?im)balance\s+due[:\s]*([\d,]+\.?\d{0,2})",
                          r"(?im)total\s+due[:\s]*([\d,]+\.?\d{0,2})",
                          r"(?im)net\s+amount[:\s]*([\d,]+\.?\d{0,2})",
                          r"(?im)invoice\s+total[:\s]*([\d,]+\.?\d{0,2})"]
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            ls = line.strip()
            if not ls:
                continue
            for p in total_patterns:
                m = re.search(p, ls)
                if m:
                    num = m.group(1).replace(',', '').replace(' ', '')
                    try:
                        detected_total = float(num)
                        detected_total_line = ls
                    except Exception:
                        continue
            # Handle label-only lines where label is on one line and amount on the next line(s)
            # e.g. a line with 'Total' followed by a next line with the numeric amount
            label_only_match = re.match(r'(?im)^(total|amount\s+due|balance\s+due|total\s+due|net\s+amount|invoice\s+total|amount\s+payable)[:\-\s]*$', ls)
            if label_only_match:
                # look ahead a few lines for the amount token
                number_pattern = r'([-+]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?)'
                for j in range(idx + 1, min(idx + 4, len(lines))):
                    nxt = lines[j].strip()
                    if not nxt:
                        continue
                    mnum = re.search(number_pattern, nxt)
                    if mnum:
                        num = mnum.group(1).replace(',', '').replace(' ', '')
                        try:
                            detected_total = float(num)
                            detected_total_line = nxt
                        except Exception:
                            pass
                        break
        # Keep last occurrence if multiple
    except Exception:
        detected_total = None
        detected_total_line = None

    # Decide final summary total with priority: detected_total > computed_total > loose total_amount (from _find_total)
    final_total = None
    if detected_total is not None:
        final_total = round(float(detected_total), 2)
    elif computed_total is not None and computed_total > 0:
        final_total = round(float(computed_total), 2)
    elif total_amount is not None:
        try:
            final_total = round(float(total_amount), 2)
        except Exception:
            final_total = None

    # Fill template where possible (invoice details + chosen total)
    try:
        template["invoice_details"]["invoice_number"] = invoice_number or None
        template["invoice_details"]["invoice_date"] = invoice_date or None
        if final_total is not None:
            template.setdefault("summary", {})
            template["summary"]["total_amount"] = final_total
        else:
            template.setdefault("summary", {})
            template["summary"]["total_amount"] = None
    except Exception:
        pass

    # Codes is preserved (local extractor does not decode barcodes here)
    template["codes"] = template.get("codes", [])
    # Attach debug info so callers can inspect OCR raw text and confidences
    template["_debug"] = {
        "raw_text": text,
        "ocr_avg_confidence": debug.get("avg_confidence"),
        "best_ocr_variant": debug.get("best_ocr_variant"),
        "ocr_words_sample": debug.get("words", [])[:40],
        "parsed_line_items_count": len(parsed_items) if parsed_items else 0,
        "parsed_line_items_sample": (parsed_items[:20] if parsed_items else []),
        "computed_total": computed_total,
        "detected_total": detected_total,
        "detected_total_line": detected_total_line
    }
    return template


def parse_raw_text_to_json(raw_text: str) -> Dict[str, Any]:
    """Heuristic conversion of raw OCR text (string) into the invoice JSON template.

    Useful when you already have OCR raw text and want to convert it to the schema without re-running OCR on images.
    """
    template = _load_schema_template()
    text = _normalize_ocr_text(raw_text or "")
    invoice_number = _find_invoice_number(text)
    invoice_date = _find_date(text)
    total_amount = _find_total(text)

    # reuse internal parser for lines
    # define lightweight _parse_line_items here similar to the function used above
    def _parse_line_items_simple(raw: str):
        items = []
        lines = [l for l in raw.splitlines() if l.strip()]
        decimal_re = re.compile(r"[\d]{1,3}(?:[\d,]*\d)?(?:\.\d{1,2})")
        int_re = re.compile(r"\b\d{1,5}\b")
        for ln in lines:
            # find last numeric token as amount
            decs = decimal_re.findall(ln)
            decs = [d.replace(',', '') for d in decs]
            amount = None
            unit_price = None
            if decs:
                try:
                    amount = float(decs[-1])
                except Exception:
                    amount = None
                if len(decs) >= 2:
                    try:
                        unit_price = float(decs[-2])
                    except Exception:
                        unit_price = None

            # try to extract a qty as any integer token
            qty = None
            ints = int_re.findall(ln)
            for t in ints:
                try:
                    ti = int(t)
                    if 0 < ti <= 100000:
                        qty = ti
                        break
                except Exception:
                    continue

            # product name: text before first decimal token
            desc = ln
            m = decimal_re.search(ln)
            if m:
                desc = ln[:m.start()]
            desc = re.sub(r"[^\w\s\-\,\.]", " ", desc).strip()

            # very small heuristic: only include lines that contain at least one digit (amount)
            if amount is None and qty is None and unit_price is None:
                continue

            item = {
                "line_number": None,
                "prod_code": None,
                "barcode": None,
                "product_name": desc or None,
                "description": desc or None,
                "packing": None,
                "unit": None,
                "unit_of_measure": None,
                "qty": qty,
                "quantity": qty,
                "unit_price": unit_price,
                "gross_amount": None,
                "discount": None,
                "taxed": False,
                "vat_percent": None,
                "net_value": None,
                "excise": None,
                "total_incl_excise": None,
                "vat_amount": None,
                "amount": amount,
            }
            items.append(item)
        return items

    parsed_items = _parse_line_items_simple(text)
    if parsed_items:
        template["line_items"] = parsed_items
        try:
            computed_total = sum([i.get("amount") or 0.0 for i in parsed_items])
        except Exception:
            computed_total = None
    else:
        computed_total = None

    # Decide final total
    final_total = None
    if total_amount is not None:
        final_total = round(float(total_amount), 2)
    elif computed_total is not None and computed_total > 0:
        final_total = round(float(computed_total), 2)

    try:
        template["invoice_details"]["invoice_number"] = invoice_number or None
        template["invoice_details"]["invoice_date"] = invoice_date or None
        if final_total is not None:
            template.setdefault("summary", {})
            template["summary"]["total_amount"] = final_total
        else:
            template.setdefault("summary", {})
            template["summary"]["total_amount"] = None
    except Exception:
        pass

    template["_debug"] = {
        "raw_text": text,
        "parsed_line_items_count": len(parsed_items),
        "parsed_line_items_sample": parsed_items[:20],
        "computed_total": computed_total,
        "detected_total": total_amount,
    }
    return template
