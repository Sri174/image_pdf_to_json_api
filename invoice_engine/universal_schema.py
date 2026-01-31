# universal_schema.py
# This file provides the universal invoice schema as a Python dict for rule-based extraction fallback.

UNIVERSAL_SCHEMA_DICT = {
    "document_type": "invoice",
    "company": {
        "name": "",
        "address": {
            "street": "",
            "city": "",
            "state": "",
            "zip": "",
            "country": ""
        },
        "contact": {
            "phone": "",
            "email": "",
            "website": ""
        },
        "tax_id": ""
    },
    "invoice_details": {
        "invoice_number": "",
        "purchase_order_number": "",
        "invoice_date": "",
        "due_date": "",
        "payment_terms": ""
    },
    "bill_to": {
        "name": "",
        "company": "",
        "address": {
            "street": "",
            "city": "",
            "state": "",
            "zip": "",
            "country": ""
        },
        "phone": "",
        "email": "",
        "customer_id": ""
    },
    "ship_to": {
        "name": "",
        "company": "",
        "address": {
            "street": "",
            "city": "",
            "state": "",
            "zip": "",
            "country": ""
        }
    },
    "line_items": [
        {
            "line_number": None,
            "description": "",
            "quantity": None,
            "unit_of_measure": "",
            "unit_price": None,
            "discount": None,
            "taxed": False,
            "tax_rate_percent": None,
            "amount": None
        }
    ],
    "summary": {
        "subtotal": None,
        "discount_total": None,
        "taxable_amount": None,
        "tax_rate_percent": None,
        "tax_total": None,
        "shipping": None,
        "other_charges": None,
        "total_amount": None,
        "amount_paid": None,
        "balance_due": None,
        "currency": ""
    },
    "payment_instructions": {
        "payable_to": "",
        "payment_method": "",
        "bank_details": {
            "bank_name": "",
            "account_name": "",
            "account_number": "",
            "ifsc_swift": ""
        },
        "notes": []
    },
    "footer": {
        "notes": [],
        "thank_you_note": ""
    }
}
