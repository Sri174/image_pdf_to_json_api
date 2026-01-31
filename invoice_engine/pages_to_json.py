"""
pages_to_json.py
----------------
Convert OCR text extracted page-by-page into a single JSON invoice
following the user-provided schema and strict rules.

Usage:
  from invoice_engine.pages_to_json import consolidate_invoice_from_pages
  result = consolidate_invoice_from_pages(list_of_page_texts)

Rules implemented (brief):
- Treat all pages as one invoice; header/customer from page 1 only; totals from last page only.
- Line items collected from all pages with `_page` field (1-based page index).
- Do not invent values: if a field can't be found, set to None.
"""
from typing import List, Dict, Any, Optional
import re


NUM_RE = re.compile(r"[-+]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?")
INT_RE = re.compile(r"\b\d+\b")


def _first_nonempty_line(text: str) -> Optional[str]:
    for ln in text.splitlines():
        s = ln.strip()
        if s:
            return s
    return None


def _find_invoice_number(text: str) -> Optional[str]:
    patterns = [
        r"(?im)invoice\s*(?:number|no|#)[:\s]*([A-Z0-9\-\/_\.]+)",
        r"(?im)inv(?:ice)?\s*[:#]\s*([A-Z0-9\-\/_\.]+)",
        r"(?im)no\.\s*([A-Z0-9\-\/_\.]+)\b",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).strip()
    # fallback: first token like INV123 or digits of length >=4 on page 1
    m = re.search(r"\bINV\w{2,}\b", text, re.I)
    if m:
        return m.group(0).strip()
    m = re.search(r"\b\d{4,}\b", text)
    if m:
        return m.group(0)
    return None


def _find_date(text: str) -> Optional[str]:
    m = re.search(r"(\d{4}[\-]\d{1,2}[\-]\d{1,2})", text)
    if m:
        return m.group(1)
    m = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", text)
    if m:
        return m.group(1)
    return None


def _find_totals_block(text: str) -> Dict[str, Optional[float]]:
    # look for labeled totals on the page
    res = {
        'subtotal': None,
        'discount_total': None,
        'taxable_amount': None,
        'tax_rate_percent': None,
        'vat_total': None,
        'shipping': None,
        'other_charges': None,
        'total_amount': None,
        'amount_paid': None,
        'balance_due': None,
        'currency': None
    }
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # simple label->key map
    mapping = {
        'subtotal': ['subtotal', 'sub total'],
        'discount_total': ['discount total', 'total discount', 'discount'],
        'taxable_amount': ['taxable amount'],
        'vat_total': ['vat', 'vat total', 'tax total'],
        'shipping': ['shipping', 'freight'],
        'other_charges': ['other charges'],
        'total_amount': ['total amount', 'amount due', 'balance due', 'grand total', 'invoice total', 'total payable'],
        'amount_paid': ['amount paid', 'paid'],
        'balance_due': ['balance due', 'amount due']
    }

    for ln in lines[::-1]:  # search bottom-up (last occurrence preferred)
        low = ln.lower()
        for key, tags in mapping.items():
            for t in tags:
                if t in low:
                    m = NUM_RE.search(ln)
                    if m:
                        try:
                            val = float(m.group(0).replace(',', '').replace(' ', ''))
                        except Exception:
                            val = None
                        # assign if not already present (we search bottom-up so last wins)
                        if res.get(key) is None:
                            res[key] = val
        # currency detection
        if res.get('currency') is None:
            cur = re.search(r"\b(AED|USD|EUR|GBP|SAR|AED)\b", ln)
            if cur:
                res['currency'] = cur.group(0)
    return res


def _parse_line_items_from_pages(pages: List[str]) -> List[Dict[str, Any]]:
    items = []
    seen_hashes = set()
    for p_idx, page in enumerate(pages, start=1):
        lines = [l.strip() for l in page.splitlines() if l.strip()]
        for ln in lines:
            # a heuristic: line must contain at least one numeric token (amount) and some text
            nums = NUM_RE.findall(ln)
            if not nums:
                continue
            # attempt to extract product code (12-13 digit) or alphanumeric tokens
            code_m = re.search(r"\b(\d{12,13})\b", ln)
            prod_code = None
            if code_m:
                prod_code = code_m.group(1)
            # extract amounts: rightmost numeric token is amount
            toks = [t for t in nums]
            amt_raw = toks[-1] if toks else None
            amt = None
            if amt_raw:
                try:
                    amt = float(amt_raw.replace(',', '').replace(' ', ''))
                except Exception:
                    amt = None
            unit_price = None
            qty = None
            if len(toks) >= 2:
                # second last candidate for unit price
                try:
                    unit_price = float(toks[-2].replace(',', '').replace(' ', ''))
                except Exception:
                    unit_price = None
            if len(toks) >= 3:
                try:
                    qtmp = toks[-3].replace(',', '').replace(' ', '')
                    if qtmp.isdigit():
                        qty = int(qtmp)
                    else:
                        qty = float(qtmp)
                except Exception:
                    qty = None

            # product name: text before first numeric token occurrence
            first_num = re.search(NUM_RE.pattern, ln)
            prod_name = None
            if first_num:
                prod_name = ln[:first_num.start()].strip()
            else:
                prod_name = ln

            # Build item preserving prod_code/barcode strings
            item = {
                'line_number': None,
                'prod_code': prod_code if prod_code is not None else None,
                'barcode': prod_code if prod_code is not None else None,
                'product_name': prod_name if prod_name else None,
                'description': prod_name if prod_name else None,
                'packing': None,
                'unit': None,
                'unit_of_measure': None,
                'qty': qty,
                'quantity': qty,
                'unit_price': unit_price,
                'gross_amount': None,
                'discount': None,
                'taxed': False,
                'vat_percent': None,
                'net_value': None,
                'excise': None,
                'total_incl_excise': None,
                'vat_amount': None,
                'amount': amt,
                '_page': p_idx
            }
            # avoid duplicates: hash by prod_name+amt+page
            h = (item.get('product_name') or '') + '||' + str(item.get('amount')) + '||' + str(p_idx)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            items.append(item)
    return items


def consolidate_invoice_from_pages(pages: List[str]) -> Dict[str, Any]:
    """Main entrypoint. `pages` is a list of OCR text strings, page order preserved.

    Returns a JSON-serializable dict matching the requested schema. Fields not
    found are set to None (null in JSON). All pages treated as one invoice.
    """
    result: Dict[str, Any] = {}

    # Header: vendor and customer only from page 1
    page1 = pages[0] if pages else ""
    vendor_company = _first_nonempty_line(page1)
    invoice_num = _find_invoice_number(page1)
    invoice_date = _find_date(page1)

    header = {
        'vendor_details': {
            'company_name_en': vendor_company or None,
            'company_name_ar': None,
            'address': None,
            'contact_info': {
                'head_office_tel': None,
                'head_office_fax': None,
                'showroom_tel': None,
                'showroom_fax': None,
                'email': None
            },
            'tax_registration_number': None
        },
        'invoice_details': {
            'invoice_number': invoice_num or None,
            'invoice_date': invoice_date or None,
            'invoice_type': None,
            'order_number': None,
            'order_date': None,
            'page_number': 1,
            'purchase_order_number': None,
            'due_date': None,
            'payment_terms': None,
            'personnel': {
                'sales_person': None,
                'supervisor': None,
                'merchandiser': None
            }
        },
        'customer_details': {
            'customer_code': None,
            'name': None,
            'address': None,
            'phone': None,
            'email': None,
            'trn': None
        }
    }

    # Company / bill_to / ship_to left as None per rule unless present on page1
    company = {
        'name': None,
        'address': {'street': None, 'city': None, 'state': None, 'zip': None, 'country': None},
        'contact': {'phone': None, 'email': None, 'website': None},
        'tax_id': None
    }
    bill_to = {'name': None, 'company': None, 'address': {'street': None, 'city': None, 'state': None, 'zip': None, 'country': None}, 'phone': None, 'email': None, 'customer_id': None}
    ship_to = {'name': None, 'company': None, 'address': {'street': None, 'city': None, 'state': None, 'zip': None, 'country': None}}

    # Invoice-level details (top-level) copy basic invoice_number/date if found
    invoice_details = {'invoice_number': invoice_num or None, 'purchase_order_number': None, 'invoice_date': invoice_date or None, 'due_date': None, 'payment_terms': None}

    # Line items: from all pages
    line_items = _parse_line_items_from_pages(pages)

    # Summary/totals: only from last page
    last_page = pages[-1] if pages else ""
    totals = _find_totals_block(last_page)

    summary = {
        'subtotal': totals.get('subtotal'),
        'discount_total': totals.get('discount_total'),
        'taxable_amount': totals.get('taxable_amount'),
        'tax_rate_percent': totals.get('tax_rate_percent'),
        'vat_total': totals.get('vat_total'),
        'shipping': totals.get('shipping'),
        'other_charges': totals.get('other_charges'),
        'total_amount': totals.get('total_amount'),
        'amount_paid': totals.get('amount_paid'),
        'balance_due': totals.get('balance_due'),
        'currency': totals.get('currency') or None
    }

    payment_instructions = {'payable_to': None, 'payment_method': None, 'bank_details': {'bank_name': None, 'account_name': None, 'account_number': None, 'ifsc_swift': None}, 'notes': []}
    codes = []
    footer = {'totals_summary': {'total_discount': None, 'total_net_inv_value': None, 'total_excise': None, 'total_incl_excise': None, 'total_vat_aed': None, 'total_incl_vat_aed': None}, 'remarks_and_notes': {'rebate_note': None, 'payment_terms': None, 'return_policy': None, 'delivery_remarks': None}, 'processing_info': {'prepared_by': None, 'printed_by': None, 'print_timestamp': None, 'warehouse_loc': None}, 'notes': [], 'thank_you_note': None}

    consolidated = {
        'header': header,
        'document_type': 'invoice',
        'company': company,
        'bill_to': bill_to,
        'ship_to': ship_to,
        'invoice_details': invoice_details,
        'line_items': line_items,
        'summary': summary,
        'payment_instructions': payment_instructions,
        'codes': codes,
        'footer': footer
    }

    return consolidated
