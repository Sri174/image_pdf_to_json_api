"""
orchestrator.py
---------------
Main pipeline orchestrator for invoice processing.
- Handles file ingestion, PDF/image conversion
- Calls DONUT inference module
- Passes output to validation and SAP prep logic
"""

import os
from PIL import Image
from typing import Union
from invoice_engine.donut_inference import DonutInvoiceExtractor

# Universal schema and task prompt (as per requirements)
UNIVERSAL_SCHEMA = '''{
  "document_type": "invoice",
  "company": {"name": "", "address": {"street": "", "city": "", "state": "", "zip": "", "country": ""}, "contact": {"phone": "", "email": "", "website": ""}, "tax_id": ""},
  "invoice_details": {"invoice_number": "", "purchase_order_number": "", "invoice_date": "", "due_date": "", "payment_terms": ""},
  "bill_to": {"name": "", "company": "", "address": {"street": "", "city": "", "state": "", "zip": "", "country": ""}, "phone": "", "email": "", "customer_id": ""},
  "ship_to": {"name": "", "company": "", "address": {"street": "", "city": "", "state": "", "zip": "", "country": ""}},
  "line_items": [{"line_number": null, "description": "", "quantity": null, "unit_of_measure": "", "unit_price": null, "discount": null, "taxed": false, "tax_rate_percent": null, "amount": null}],
  "summary": {"subtotal": null, "discount_total": null, "taxable_amount": null, "tax_rate_percent": null, "tax_total": null, "shipping": null, "other_charges": null, "total_amount": null, "amount_paid": null, "balance_due": null, "currency": ""},
  "payment_instructions": {"payable_to": "", "payment_method": "", "bank_details": {"bank_name": "", "account_name": "", "account_number": "", "ifsc_swift": ""}, "notes": []},
  "footer": {"notes": [], "thank_you_note": ""}
}'''

DONUT_TASK_PROMPT = (
    "Extract all invoice information from this document and return the output "
    "STRICTLY in the following JSON format. Do not change the structure. "
    "If a value is not present, return null or an empty string.\n" + UNIVERSAL_SCHEMA
)

def load_image_from_file(file_path: str) -> Image.Image:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        return Image.open(file_path)
    elif ext == ".pdf":
        from pdf2image import convert_from_path
        images = convert_from_path(
            file_path,
            poppler_path=r"C:\Program Files\poppler-25.12.0\Library\bin"
        )
        return images[0]  # Only first page for now
    else:
        raise ValueError(f"Unsupported file type: {ext}")

class InvoiceProcessingOrchestrator:
    def __init__(self, donut_model_path: str):
        self.donut = DonutInvoiceExtractor(donut_model_path)

    def process_invoice(self, file_path: str) -> dict:
        image = load_image_from_file(file_path)
        # Strict prompt/schema validation
        expected_prompt = (
            "Extract all invoice information from this document and return the output "
            "STRICTLY in the following JSON format. Do not change the structure. "
            "If a value is not present, return null or an empty string.\n" + UNIVERSAL_SCHEMA
        )
        if DONUT_TASK_PROMPT.strip() != expected_prompt.strip():
            raise ValueError("DONUT task prompt does not match the required universal schema. This is a critical error.")
        donut_result = self.donut.extract_invoice(image, DONUT_TASK_PROMPT)
        # Post-processing, validation, SAP prep to be added
        return donut_result
