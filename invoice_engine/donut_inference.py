"""
donut_inference.py
------------------
DONUT model inference module for universal invoice extraction.
- Loads DONUT at startup
- Accepts image and task prompt
- Returns raw JSON output
- No business logic or schema enforcement here
"""

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json

class DonutInvoiceExtractor:
    def __init__(self, model_name_or_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Use local model files, ensure both processor and model are loaded from the same directory
        self.model_dir = model_name_or_path
        self.processor = DonutProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.model_dir,
            local_files_only=True,
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.model.eval()

    def extract_invoice(self, image: Image.Image, task_prompt: str) -> dict:
        # PHASE 1: DISABLE DONUT, USE SIMPLE OCR + RULES
        import re
        import pytesseract
        from invoice_engine.universal_schema import UNIVERSAL_SCHEMA_DICT
        from copy import deepcopy

        if image is None:
            return {"error": "Input image is None"}
        if image.mode != "RGB":
            image = image.convert("RGB")

        # OCR
        ocr_text = pytesseract.image_to_string(image)
        print("[DEBUG] Raw OCR output:\n" + ocr_text)

        # Simple regex/keyword extraction
        def extract_field(pattern, text, default=None):
            match = re.search(pattern, text, re.IGNORECASE)
            return match.group(1).strip() if match else default

        invoice_json = deepcopy(UNIVERSAL_SCHEMA_DICT)
        invoice_json["invoice_details"]["invoice_number"] = extract_field(r"INVOICE NO[:\-]?\s*(\S+)", ocr_text, "")
        invoice_json["invoice_details"]["invoice_date"] = extract_field(r"DATE[:\-]?\s*([\d\-/]+)", ocr_text, "")
        invoice_json["company"]["name"] = extract_field(r"Your Company Name(.*)", ocr_text, "Your Company Name")
        invoice_json["summary"]["total_amount"] = extract_field(r"Balance Due\s*\$\s*([\d,.]+)", ocr_text, None)
        invoice_json["summary"]["subtotal"] = extract_field(r"SUBTOTAL\s*\$\s*([\d,.]+)", ocr_text, None)
        invoice_json["summary"]["discount_total"] = extract_field(r"DISCOUNT\s*\$\s*([\d,.]+)", ocr_text, None)
        invoice_json["summary"]["tax_rate_percent"] = extract_field(r"TAX RATE\s*([\d.]+)%", ocr_text, None)
        invoice_json["summary"]["tax_total"] = extract_field(r"TOTAL TAX\s*\$\s*([\d,.]+)", ocr_text, None)
        invoice_json["summary"]["shipping"] = extract_field(r"SHIPPING/HANDLING\s*\$\s*([\d,.]+)", ocr_text, None)
        # Line items: very basic, just count lines with qty/price
        line_items = []
        for line in ocr_text.splitlines():
            m = re.match(r"^(\w[\w\s]*)\s+(\d+)\s+\$\s*([\d.]+)\s+\$\s*([\d.]+)", line)
            if m:
                line_items.append({
                    "description": m.group(1).strip(),
                    "quantity": int(m.group(2)),
                    "unit_price": float(m.group(3)),
                    "amount": float(m.group(4)),
                    "line_number": None,
                    "unit_of_measure": "",
                    "discount": None,
                    "taxed": False,
                    "tax_rate_percent": None
                })
        if line_items:
            invoice_json["line_items"] = line_items

        return invoice_json
