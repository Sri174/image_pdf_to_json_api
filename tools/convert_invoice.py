#!/usr/bin/env python3
"""CLI helper: convert a PDF or image to the unified invoice JSON using local parsers.
Usage: python tools/convert_invoice.py <input_path>
Saves result to runs/<timestamp>_converted.json and prints path.
"""
import sys, os, json
from datetime import datetime
from invoice_engine.multipage_parser import parse_multipage_invoice
from invoice_engine.local_extraction import local_extract_invoice, parse_raw_text_to_json

INPUT = sys.argv[1] if len(sys.argv) > 1 else None
if not INPUT:
    print("Usage: python tools/convert_invoice.py <input_path>")
    sys.exit(1)

os.makedirs('runs', exist_ok=True)

def try_pdf(path):
    # try multipage parser first
    try:
        res = parse_multipage_invoice(path)
        if res and res.get('line_items'):
            return res
    except Exception:
        pass
    # fallback: convert pages to images and run auto-crop image sweep
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(path, poppler_path=r"C:\Program Files\poppler-25.12.0\Library\bin")
    except Exception:
        images = []
    for idx, img in enumerate(images):
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format='JPEG')
        b = buf.getvalue()
        try:
            r = local_extract_invoice(b)
            # if only raw_text, convert
            if (not r.get('line_items')) and r.get('_debug', {}).get('raw_text'):
                conv = parse_raw_text_to_json(r['_debug']['raw_text'])
                return conv
            if r and r.get('line_items'):
                return r
        except Exception:
            continue
    return {'line_items': [], 'summary': {'total_amount': 0}, '_debug': {'note': 'no_result'}}

def try_image(path):
    with open(path, 'rb') as f:
        b = f.read()
    r = local_extract_invoice(b)
    if (not r.get('line_items')) and r.get('_debug', {}).get('raw_text'):
        return parse_raw_text_to_json(r['_debug']['raw_text'])
    return r

inp = INPUT
if not os.path.exists(inp):
    print('Input not found:', inp)
    sys.exit(2)

ext = os.path.splitext(inp)[1].lower()
if ext == '.pdf':
    out = try_pdf(inp)
else:
    out = try_image(inp)

fname = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ_') + os.path.basename(inp).replace(' ', '_') + '_converted.json'
outpath = os.path.join('runs', fname)
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('Saved:', outpath)
print(json.dumps(out, indent=2))
