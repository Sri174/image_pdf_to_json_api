"""
sample_run.py
-------------
Sample execution of the invoice processing pipeline.
- Uses dummy invoice file and mock existing invoices
- Prints final SAP payload
"""

from invoice_engine.orchestrator import InvoiceProcessingOrchestrator
from invoice_engine.validation import validate_invoice_json
from invoice_engine.duplicate_check import is_duplicate
from invoice_engine.sap_prep import prepare_sap_payload

# Dummy config and data
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")  # Local model directory
DUMMY_INVOICE_FILE = "sample_invoice.pdf"  # Place a sample file in workspace for real test
EXISTING_INVOICES = [
    {
        "company": {"tax_id": "123456789"},
        "invoice_details": {"invoice_number": "INV-001", "invoice_date": "2024-01-01"},
        "summary": {"total_amount": 1000.0}
    }
]

if __name__ == "__main__":
    orchestrator = InvoiceProcessingOrchestrator(MODEL_PATH)
    donut_json = orchestrator.process_invoice(DUMMY_INVOICE_FILE)
    validation = validate_invoice_json(donut_json)
    duplicate = is_duplicate(donut_json, EXISTING_INVOICES)
    status = "READY_FOR_UPLOAD"
    if validation["status"] != "READY_FOR_UPLOAD":
        status = "NEEDS_REVIEW"
    if duplicate["duplicate"]:
        status = "DUPLICATE_BLOCKED"
    sap_payload = prepare_sap_payload(donut_json, DUMMY_INVOICE_FILE, status)
    import json
    print(json.dumps(sap_payload, indent=2))
