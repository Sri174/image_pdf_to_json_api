"""
Simple demo to test the /ocr endpoint and see console logs
"""
import subprocess
import time
import requests
import json
import os

print("="*60)
print("POST /ocr Endpoint Demo")
print("="*60)

# Check if GEMINI_API_KEY is set
gemini_key = os.getenv("GEMINI_API_KEY")
print(f"\nGEMINI_API_KEY status: {'✓ SET (Gemini will be used)' if gemini_key else '✗ NOT SET (will use local OCR)'}")

# Check if server is running
url = "http://127.0.0.1:8000/health"
try:
    response = requests.get(url, timeout=2)
    if response.status_code == 200:
        print("✓ Server is running")
    else:
        print("⚠ Server returned unexpected status")
except Exception:
    print("✗ Server is NOT running!")
    print("\nPlease start it in another terminal with:")
    print("  uvicorn api_server:app --reload")
    print("\nYou'll see console logs like:")
    print("  [OCR] Processing file: <filename>")
    print("  [OCR] Gemini API key present: True/False")
    print("  [OCR] Attempting Gemini extraction... (if key is set)")
    print("  [OCR] Using local OCR extraction... (fallback)")
    exit(1)

# Test the OCR endpoint
test_file = r"runs\20260202T044111Z_7e0e6d89_1004 gormet.pdf"

if not os.path.exists(test_file):
    print(f"\n✗ Test file not found: {test_file}")
    exit(1)

print(f"\nTesting: {test_file}")
print("-"*60)

try:
    with open(test_file, "rb") as f:
        files = {"file": (os.path.basename(test_file), f, "application/pdf")}
        print("Sending POST request to /ocr...")
        response = requests.post("http://127.0.0.1:8000/ocr", files=files, timeout=60)
    
    if response.status_code == 200:
        print(f"✓ Status: {response.status_code} OK")
        result = response.json()
        
        # Show key info
        print(f"\nExtraction completed:")
        print(f"  - Barcodes found: {len(result.get('codes', []))}")
        print(f"  - Line items: {len(result.get('line_items', []))}")
        print(f"  - Has debug info: {'✓' if '_debug' in result else '✗'}")
        
        if '_debug' in result:
            debug = result['_debug']
            print(f"  - OCR confidence: {debug.get('ocr_avg_confidence', 0):.1f}%")
            print(f"  - OCR words detected: {debug.get('best_ocr_variant', {}).get('n_words', 0)}")
        
        print(f"\n{'='*60}")
        print("IMPORTANT: Check the server terminal for console logs!")
        print("You should see lines like:")
        print("  [OCR] Processing file: ...")
        print("  [OCR] Gemini API key present: True/False")
        print("  [OCR] Using local OCR extraction...")
        print("="*60)
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"✗ Request failed: {e}")
