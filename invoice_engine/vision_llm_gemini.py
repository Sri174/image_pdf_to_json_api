def list_gemini_models():
    """List available Gemini models for this API key."""
    url = "https://generativelanguage.googleapis.com/v1/models"
    params = {"key": GEMINI_API_KEY}
    resp = requests.get(url, params=params)
    print("[Gemini ListModels Response]")
    print(json.dumps(resp.json(), indent=2))
# vision_llm_gemini.py
# Vision LLM invoice extraction using Google Gemini API

import requests
import json
import base64
import os
import re
import time
from typing import List, Dict, Any, Optional, Union
import io
from PIL import Image, ImageFilter, ImageOps
import statistics
import pytesseract
from requests.exceptions import RequestException

# Optional: support Google Application Default Credentials for OAuth access token
try:
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest
except Exception:
    google = None

# Fields we consider mandatory for a sensible invoice extraction
REQUIRED_FIELD_PATHS = [
    ("header", "invoice_details", "invoice_number"),
    ("header", "invoice_details", "invoice_date"),
    ("summary", "total_amount"),
]

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
# Read API key from environment where possible; fall back to hardcoded (not recommended)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


SYSTEM_PROMPT = (
    "You are an AI specialized in understanding invoice documents. "
    "You mustP return ONLY a valid JSON object that matches the schema provided below, with no extra or missing fields. "
    "Use the schema as a template: every key and structure must be present in your output, even if values are null or empty. "
    "Do not include markdown, code blocks, or any text outside the JSON. "
    "Do not explain, comment, or format the output. "
    "Return only the JSON object, no explanations or formatting. "
    "If a value is not present in the invoice, return null or an empty string as appropriate. "
    "Do not add or remove fields. Extract line items accurately. "
    "STRICTLY FOLLOW THE SCHEMA STRUCTURE."
)
USER_PROMPT = "Analyze the attached invoice image and return the JSON using the exact schema below."

with open("invoice_engine/universal_schema.json", "r") as f:
    UNIVERSAL_SCHEMA = f.read()


def extract_invoice_with_gemini(image_bytes: Union[bytes, List[bytes]]) -> dict:
    """Send one or more page images to Gemini and return parsed JSON.

    Accepts a single image bytes or a list of image bytes (for multi-page PDFs).
    """
    parts = []
    # first part: instructions + schema
    parts.append({"text": SYSTEM_PROMPT + "\n" + USER_PROMPT + "\n" + UNIVERSAL_SCHEMA})

    # normalize to list
    imgs: List[bytes] = image_bytes if isinstance(image_bytes, list) else [image_bytes]
    for img in imgs:
        # Gemini expects base64-encoded inline_data
        image_b64 = base64.b64encode(img).decode("utf-8")
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})

    payload = {"contents": [{"role": "user", "parts": parts}]}
    # Prepare auth headers: prefer OAuth via ADC; if absent, fall back to API key as query param
    # Call local helper to get headers (may attach OAuth token via ADC)
    try:
        headers = _get_gemini_auth_headers()
    except Exception:
        headers = {"Content-Type": "application/json"}

    # If ADC didn't provide Authorization header, include API key as query param (if present)
    params = {}
    if "Authorization" not in headers and GEMINI_API_KEY:
        params = {"key": GEMINI_API_KEY}

    try:
        response = _post_with_retries(GEMINI_API_URL, headers=headers, params=params, data=json.dumps(payload), timeout=60, retries=4)
    except RequestException as e:
        return {
            "status": "NEEDS_REVIEW",
            "error": f"Vision LLM request failed: {e}",
            "raw_response": None,
            "_gemini_diagnostics": {"exception": str(e)}
        }

    try:
        # Try to parse JSON, but be tolerant to different response shapes
        try:
            result = response.json()
        except Exception:
            # Not JSON - return raw text for diagnostics
            return {
                "status": "NEEDS_REVIEW",
                "error": "Gemini returned non-JSON response",
                "raw_response": response.text,
                "_gemini_diagnostics": {"status_code": response.status_code}
            }

        # Locate text in known return formats
        text = None
        try:
            # Primary: candidates -> content -> parts -> text
            text = result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            # Fallbacks
            if isinstance(result.get("output"), str):
                text = result.get("output")
            elif isinstance(result.get("text"), str):
                text = result.get("text")
            else:
                # try nested alternatives
                try:
                    choices = result.get("choices") or result.get("outputs")
                    if isinstance(choices, list) and choices:
                        first = choices[0]
                        if isinstance(first, dict):
                            for k in ("text", "content", "message"):
                                if k in first and isinstance(first[k], str):
                                    text = first[k]
                                    break
                except Exception:
                    text = None
        # Remove markdown code block if present
        if text.strip().startswith("```"):
            text = text.strip()
            # Remove the first line (```json or ```)
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove the last line if it's ```
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)
        if not text:
            return {
                "status": "NEEDS_REVIEW",
                "error": "Gemini response contained no textual output",
                "raw_response": json.dumps(result, ensure_ascii=False),
                "_gemini_diagnostics": {"parsed_json_keys": list(result.keys())}
            }

        # Parse as JSON
        try:
            result_json = json.loads(text)
        except Exception as e:
            return {
                "status": "NEEDS_REVIEW",
                "error": f"Failed to parse Gemini text as JSON: {e}",
                "raw_response": text,
                "_gemini_diagnostics": {"response_json": result}
            }
        # Optionally, enforce schema structure here (auto-coercion/validation)
        try:
            # Attempt to fill any missing required fields by asking Gemini to re-check the OCR texts
            _ensure_required_fields_from_images(result_json, imgs)
        except Exception:
            pass
        return result_json
    except Exception as e:
        # Catch-all diagnostics
        return {
            "status": "NEEDS_REVIEW",
            "error": f"Failed to process Gemini response: {e}",
            "raw_response": getattr(response, "text", None),
            "_gemini_diagnostics": {"exception": str(e)}
        }


def call_gemini_with_text(prompt: str, timeout: int = 60) -> Optional[str]:
    """Call Gemini with a text prompt and return the assistant text output."""
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}] } ] }
    # prefer ADC OAuth header if available
    try:
        headers = _get_gemini_auth_headers()
    except Exception:
        headers = {"Content-Type": "application/json"}
    params = {}
    if "Authorization" not in headers and GEMINI_API_KEY:
        params = {"key": GEMINI_API_KEY}
    try:
        resp = _post_with_retries(GEMINI_API_URL, headers=headers, params=params, data=json.dumps(payload), timeout=timeout, retries=3)
    except RequestException:
        return None
    try:
        j = resp.json()
        text = None
        # Gemini returns candidates[0].content.parts[0].text
        if isinstance(j, dict):
            try:
                text = j["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                # fallback: try other common keys
                text = j.get("output") or j.get("text") or None
        return _strip_codefences(text) if isinstance(text, str) else None
    except Exception:
        return None


def parse_ocr_pages_with_gemini(ocr_texts: List[str]) -> Dict[str, Any]:
    """Send page-wise OCR text to Gemini and parse the returned JSON.

    Returns parsed JSON dict or raises if parsing fails.
    """
    pages_joined = []
    for i, t in enumerate(ocr_texts, start=1):
        pages_joined.append(f"-- PAGE {i} --\n" + (t or ""))
    prompt = (
        SYSTEM_PROMPT + "\n\n" + "You will receive OCR output for every page below. FOLLOW THESE RULES EXACTLY: Return ONLY a single JSON object matching the schema provided, with no extra text. Use null for missing values.\n\n" + UNIVERSAL_SCHEMA + "\n\nOCR PAGES:\n" + "\n\n".join(pages_joined)
    )
    out_text = call_gemini_with_text(prompt)
    if not out_text:
        raise RuntimeError("Empty response from Gemini")
    try:
        parsed = json.loads(out_text)
        return parsed
    except Exception as e:
        raise RuntimeError(f"Failed to parse Gemini JSON output: {e}")


def ocr_images_best_texts(image_bytes_list: List[bytes], lang: Optional[str] = None) -> List[str]:
    """Perform OCR on a list of images with several preprocessing variants and return best text per image.

    For each image, try multiple preprocessing steps (original, grayscale, resized, sharpened, threshold)
    and select the variant with highest mean OCR confidence as reported by `pytesseract.image_to_data`.
    Returns list of extracted texts (one per input image), preserving order.
    """
    results: List[str] = []
    for b in image_bytes_list:
        try:
            img = Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            results.append("")
            continue

        variants = []

        # Original
        variants.append((img, "orig"))

        # Grayscale
        variants.append((ImageOps.grayscale(img), "gray"))

        # Resized (2x)
        w, h = img.size
        variants.append((img.resize((w * 2, h * 2), resample=Image.BICUBIC), "resized"))

        # Sharpen
        try:
            variants.append((img.filter(ImageFilter.SHARPEN), "sharpen"))
        except Exception:
            pass

        # Grayscale + threshold
        try:
            g = ImageOps.grayscale(img).point(lambda x: 0 if x < 128 else 255, mode="1")
            variants.append((g, "thresh"))
        except Exception:
            pass

        best_text = ""
        best_score = -1.0

        for var_img, name in variants:
            try:
                config = "--oem 1 --psm 3"
                data = pytesseract.image_to_data(var_img, lang=lang or None, config=config, output_type=pytesseract.Output.DICT)
                confs = []
                for c in data.get("conf", []):
                    try:
                        ci = float(c)
                        if ci >= 0:
                            confs.append(ci)
                    except Exception:
                        continue
                mean_conf = statistics.mean(confs) if confs else 0.0
                text = pytesseract.image_to_string(var_img, lang=lang or None, config=config)
                # score by mean_conf and length
                score = mean_conf * (1 + min(len(text), 200) / 200.0)
                if score > best_score:
                    best_score = score
                    best_text = text
            except Exception:
                continue

        results.append(best_text or "")

    return results


def _get_nested(d: Dict[str, Any], path: tuple):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _set_nested(d: Dict[str, Any], path: tuple, value: Any):
    cur = d
    for p in path[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[path[-1]] = value


def _ensure_required_fields_from_images(result_json: Dict[str, Any], image_bytes_list: List[bytes]):
    """Check required fields and, if missing, ask Gemini to extract them from OCR text.

    This function performs OCR locally to produce page text snippets that are included
    in a focused follow-up prompt asking Gemini to return a JSON object with only the
    missing fields filled using the exact strings found in the invoice images.
    """
    if not isinstance(result_json, dict):
        return
    missing = []
    for path in REQUIRED_FIELD_PATHS:
        v = _get_nested(result_json, path)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            missing.append(path)

    if not missing:
        return

    # Produce OCR texts for pages (best variants) to include in the follow-up prompt
    try:
        ocr_texts = ocr_images_best_texts(image_bytes_list)
    except Exception:
        ocr_texts = []

    # Build a focused follow-up prompt requesting only the missing fields
    missing_keys = [".".join(p) for p in missing]
    prompt_lines = [
        "The previous full-invoice JSON extraction is missing or has empty values for these keys:",
        ", ".join(missing_keys),
        "\nBelow are the OCR-extracted page texts from the invoice (PAGE 1 first).",
    ]
    for i, t in enumerate(ocr_texts, start=1):
        snippet = t.strip()[:2000] if isinstance(t, str) else ""
        prompt_lines.append(f"-- PAGE {i} --\n{snippet}")

    prompt_lines.append("\nTask: For each missing key, find the exact value as it appears in the invoice text above and return a JSON object with only those keys and their values. Use null if you cannot find a value. RETURN ONLY A SINGLE JSON OBJECT, NO EXPLANATION.")
    prompt = "\n\n".join(prompt_lines)

    try:
        resp = call_gemini_with_text(prompt, timeout=120)
        if not resp:
            return
        parsed = None
        try:
            parsed = json.loads(resp)
        except Exception:
            parsed = None
        if not isinstance(parsed, dict):
            return
        # Update result_json with any found values (only for the requested paths)
        for path in missing:
            key = ".".join(path)
            if key in parsed and parsed[key] is not None:
                _set_nested(result_json, path, parsed[key])
    except Exception:
        # on any error, do not raise — leave original result_json as-is
        return


def _post_with_retries(url: str, headers: Dict[str, str], params: Dict[str, str], data: str, timeout: int = 60, retries: int = 3, backoff_factor: float = 2.0):
    """POST helper with exponential backoff. Raises RequestException on final failure.

    - `timeout` is the initial per-request timeout in seconds.
    - `retries` is number of attempts (including first).
    - `backoff_factor` is base multiplier for sleep (sleep = backoff_factor ** (attempt-1)).
    """
    last_exc = None
    t = timeout
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, params=params, data=data, timeout=t)
            # don't raise for status here; caller will inspect response
            return resp
        except RequestException as e:
            last_exc = e
            if attempt == retries:
                # re-raise the last exception so callers can build diagnostics
                raise
            sleep_for = backoff_factor ** (attempt - 1)
            try:
                time.sleep(sleep_for)
            except Exception:
                pass
            # increase timeout for next attempt but cap it
            t = min(int(t * 1.8), 300)
    # If somehow loop completes, raise last exception
    if last_exc:
        raise last_exc
    raise RequestException("_post_with_retries: unknown error")


def _find_single(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip() if m.groups() else m.group(0).strip()


def _split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _extract_lines_table(lines: List[str]) -> List[Dict[str, Any]]:
    """Try to extract simple line-items from lines.

    This is conservative: it only extracts rows that contain at least one numeric token
    (price/amount) and keeps original string tokens (no numeric coercion).
    """
    items = []
    for ln in lines:
        # Skip common total/footer lines
        if re.search(r"^total\b|^subtotal\b|^vat\b|^grand total\b|amount in words", ln, flags=re.I):
            continue
        # Find all numeric-like tokens (may include commas, dots, currency symbols)
        nums = re.findall(r"[-+€$£₹\d][\d,\.\/]*", ln)
        if not nums:
            continue
        # Heuristic: last numeric tokens are amount/unitprice/qty
        parts = re.split(r"\s{2,}|\t", ln)
        # If split produced many parts, use them; else fall back to whitespace split
        if len(parts) == 1:
            parts = ln.split()
        # Build an item dict conservatively
        item = {
            "line_number": None,
            "prod_code": "",
            "barcode": "",
            "product_name": "",
            "description": "",
            "packing": "",
            "unit": "",
            "unit_of_measure": "",
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

        # Heuristic population: product name = leading text before first numeric token
        first_num = re.search(r"[-+€$£₹\d][\d,\.\/]*", ln)
        if first_num:
            product_part = ln[: first_num.start()].strip()
            item["product_name"] = product_part
        # Assign tail numeric tokens to amount fields, preserving original strings
        if nums:
            # last token -> amount
            item["amount"] = nums[-1]
            if len(nums) >= 2:
                item["unit_price"] = nums[-2]
            if len(nums) >= 3:
                item["quantity"] = nums[-3]
                item["qty"] = nums[-3]

        items.append(item)
    return items


def convert_ocr_to_universal_schema(ocr_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OCR PAGE-BY-PAGE JSON into the unified invoice schema.

    Rules implemented (per user instructions):
    - All pages belong to a single invoice.
    - Vendor/header/customer info only from page 1.
    - Line items merged from all pages.
    - Totals only taken from last page (no recomputation).
    - Do NOT hallucinate; missing values -> null.
    - Maintain original numeric strings as-is.
    """
    pages = ocr_doc.get("pages", [])
    if not pages:
        pages = []

    # Prepare empty template by loading the JSON schema file (keeps structure)
    try:
        schema = json.loads(UNIVERSAL_SCHEMA)
    except Exception:
        # fallback minimal structure if schema file missing
        schema = {
            "header": {
                "vendor_details": {"company_name_en": "", "company_name_ar": "", "address": "", "contact_info": {"head_office_tel": "", "head_office_fax": "", "showroom_tel": "", "showroom_fax": "", "email": ""}, "tax_registration_number": ""},
                "invoice_details": {"invoice_number": "", "invoice_date": "", "invoice_type": "", "order_number": "", "order_date": "", "page_number": "", "purchase_order_number": "", "due_date": "", "payment_terms": "", "personnel": {"sales_person": "", "supervisor": "", "merchandiser": ""}},
                "customer_details": {"customer_code": "", "name": "", "address": "", "phone": "", "email": "", "trn": ""}
            },
            "document_type": "invoice",
            "company": {"name": "", "address": {"street": "", "city": "", "state": "", "zip": "", "country": ""}, "contact": {"phone": "", "email": "", "website": ""}, "tax_id": ""},
            "bill_to": {"name": "", "company": "", "address": {"street": "", "city": "", "state": "", "zip": "", "country": ""}, "phone": "", "email": "", "customer_id": ""},
            "ship_to": {"name": "", "company": "", "address": {"street": "", "city": "", "state": "", "zip": "", "country": ""}},
            "invoice_details": {"invoice_number": "", "purchase_order_number": "", "invoice_date": "", "due_date": "", "payment_terms": ""},
            "line_items": [schema.get("line_items", [{}]) if isinstance(schema, dict) else [{}]],
            "summary": {"subtotal": None, "discount_total": None, "taxable_amount": None, "tax_rate_percent": None, "vat_total": None, "shipping": None, "other_charges": None, "total_amount": None, "amount_paid": None, "balance_due": None, "currency": ""},
            "payment_instructions": {"payable_to": "", "payment_method": "", "bank_details": {"bank_name": "", "account_name": "", "account_number": "", "ifsc_swift": ""}, "notes": []},
            "codes": [{"type": "", "value": "", "confidence": 1.0}],
            "footer": {"totals_summary": {"total_discount": None, "total_net_inv_value": None, "total_excise": None, "total_incl_excise": None, "total_vat_aed": None, "total_incl_vat_aed": None}, "remarks_and_notes": {"rebate_note": "", "payment_terms": "", "return_policy": "", "delivery_remarks": ""}, "processing_info": {"prepared_by": "", "printed_by": "", "print_timestamp": "", "warehouse_loc": ""}, "notes": [], "thank_you_note": ""}
        }

    out = schema

    # Default header values set to null where schema expects strings
    # Extract from page 1 (first page in list)
    if pages:
        p1 = pages[0]
        text1 = p1.get("text", "")
        lines1 = _split_lines(text1)
        # Vendor/company: try common labels
        company_name = None
        # Common patterns for company name: headline before "Invoice" or first non-empty line
        if lines1:
            # take first non-empty line as company name only if it doesn't look like a label
            first = lines1[0]
            if not re.search(r"invoice|tax|bill to|ship to|date|no\b", first, flags=re.I):
                company_name = first

        out["header"]["vendor_details"]["company_name_en"] = company_name or ""

        # invoice metadata
        inv_no = _find_single(r"Invoice\s*(?:No(?:\.|:)?)?\s*[:#-]?\s*(\S+)", text1) or _find_single(r"Inv(?:\.|)\s*#\s*(\S+)", text1)
        inv_date = _find_single(r"Invoice\s*Date\s*[:\-]?\s*(\S+)", text1)
        po = _find_single(r"Purchase\s*Order\s*(?:No|Number)?\s*[:\-]?\s*(\S+)", text1) or _find_single(r"PO\s*#\s*(\S+)", text1)
        due = _find_single(r"Due\s*Date\s*[:\-]?\s*(\S+)", text1)

        out["header"]["invoice_details"]["invoice_number"] = inv_no or ""
        out["header"]["invoice_details"]["invoice_date"] = inv_date or ""
        out["header"]["invoice_details"]["purchase_order_number"] = po or ""
        out["header"]["invoice_details"]["due_date"] = due or ""

        # customer details: look for "Bill To" section
        bill_text = None
        m = re.search(r"Bill\s*To[:\s]*(.+?)(?:Ship\s*To|Invoice\b|Description\b|$)", text1, flags=re.I | re.S)
        if m:
            bill_text = m.group(1).strip()
        else:
            # fallback: look for "Sold To" or "Customer"
            m2 = re.search(r"(?:Sold\s*To|Customer)[:\s]*(.+?)(?:Ship\s*To|Invoice\b|Description\b|$)", text1, flags=re.I | re.S)
            if m2:
                bill_text = m2.group(1).strip()

        if bill_text:
            # take first line as name, subsequent lines as address
            bill_lines = _split_lines(bill_text)
            out["header"]["customer_details"]["name"] = bill_lines[0] if bill_lines else ""
            out["header"]["customer_details"]["address"] = ", ".join(bill_lines[1:]) if len(bill_lines) > 1 else ""
        else:
            out["header"]["customer_details"]["name"] = ""
            out["header"]["customer_details"]["address"] = ""

        # customer phone/email/trn on page1
        phone = _find_single(r"Tel(?:ephone|)[:\s]*([\d\-\(\)\s\+]+)", text1)
        email = _find_single(r"[Ee]mail[:\s]*([\w\.-]+@\w[\w\.-]+)", text1)
        trn = _find_single(r"TRN[:\s]*([A-Z0-9\-]+)", text1) or _find_single(r"Tax\s*Registration\s*Number[:\s]*([A-Z0-9\-]+)", text1)
        out["header"]["customer_details"]["phone"] = phone or ""
        out["header"]["customer_details"]["email"] = email or ""
        out["header"]["customer_details"]["trn"] = trn or ""

    # Aggregate line items from all pages
    all_items: List[Dict[str, Any]] = []
    for pg in pages:
        text = pg.get("text", "")
        lines = _split_lines(text)
        # Try to find a table header to identify table start
        table_start_idx = None
        for i, ln in enumerate(lines):
            if re.search(r"description|qty|quantity|unit price|amount|total", ln, flags=re.I):
                table_start_idx = i + 1
                break
        table_lines = lines[table_start_idx:] if table_start_idx is not None else lines
        items = _extract_lines_table(table_lines)
        all_items.extend(items)

    # If no items found, ensure line_items is an empty list per schema
    out["line_items"] = all_items if all_items else []

    # Totals and summary from last page only
    if pages:
        last_text = pages[-1].get("text", "")
        # extract subtotal, vat, total, amount paid, balance
        def _find_total(label_patterns: List[str]) -> Optional[str]:
            for pat in label_patterns:
                m = re.search(pat + r"\s*[:\-]?\s*([\d,\.\w\s\u20AC\$£₹%-]+)", last_text, flags=re.I)
                if m:
                    return m.group(1).strip()
            return None

        subtotal = _find_total([r"Subtotal", r"Sub[- ]?total"])
        vat = _find_total([r"VAT\b", r"Tax\b"])
        total = _find_total([r"Grand\s*Total", r"Total\s*Amount", r"Amount\s*Due", r"TOTAL\b"])
        amount_paid = _find_total([r"Amount\s*Paid"])
        balance = _find_total([r"Balance\s*Due", r"Outstanding\s*Amount"])
        currency = None
        # try to find a currency code or symbol near totals
        cur_m = re.search(r"(AED|USD|EUR|GBP|SAR|QAR|OMR|KWD)\b", last_text)
        if cur_m:
            currency = cur_m.group(1)
        else:
            sym = re.search(r"([€$£₹])", last_text)
            currency = sym.group(1) if sym else None

        out["summary"]["subtotal"] = subtotal or None
        out["summary"]["vat_total"] = vat or None
        out["summary"]["total_amount"] = total or None
        out["summary"]["amount_paid"] = amount_paid or None
        out["summary"]["balance_due"] = balance or None
        out["summary"]["currency"] = currency or ""

    return out


def _strip_codefences(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return t


def _get_gemini_auth_headers() -> Dict[str, str]:
    """Return headers for Gemini requests.

    Tries to obtain OAuth2 access token via Application Default Credentials (ADC).
    If ADC available, returns Authorization header. Otherwise returns a minimal headers dict
    (Content-Type only) and callers may use the `key` query param fallback.
    """
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    try:
        if google is not None:
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            if not creds or not creds.valid:
                creds.refresh(GoogleAuthRequest())
            if creds and getattr(creds, "token", None):
                headers["Authorization"] = f"Bearer {creds.token}"
                return headers
    except Exception:
        # ADC not available or failed; fallback to no auth header
        pass
    return headers

