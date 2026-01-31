"""
duplicate_check.py
------------------
Duplicate invoice detection logic.
- Compares key fields for exact/fuzzy matches
- Flags potential duplicates
"""

from typing import Dict, Any, List
import difflib

def is_duplicate(new_inv: Dict[str, Any], existing_invoices: List[Dict[str, Any]], fuzzy: bool = True) -> dict:
    new_key = (
        (new_inv.get("company", {}).get("tax_id") or "") + "|" +
        (new_inv.get("invoice_details", {}).get("invoice_number") or "") + "|" +
        (new_inv.get("invoice_details", {}).get("invoice_date") or "") + "|" +
        str(new_inv.get("summary", {}).get("total_amount") or "")
    )
    for inv in existing_invoices:
        inv_key = (
            (inv.get("company", {}).get("tax_id") or "") + "|" +
            (inv.get("invoice_details", {}).get("invoice_number") or "") + "|" +
            (inv.get("invoice_details", {}).get("invoice_date") or "") + "|" +
            str(inv.get("summary", {}).get("total_amount") or "")
        )
        if new_key == inv_key:
            return {"duplicate": True, "type": "exact"}
        if fuzzy and difflib.SequenceMatcher(None, new_key, inv_key).ratio() > 0.95:
            return {"duplicate": True, "type": "fuzzy"}
    return {"duplicate": False}
