from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import glob
from datetime import datetime
import tempfile
from io import BytesIO

# Load .env file into environment
env_path = os.path.join(os.getcwd(), ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as _ef:
            for ln in _ef:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                if "=" not in ln:
                    continue
                k, v = ln.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and not os.getenv(k):
                    os.environ[k] = v
    except Exception:
        pass

app = FastAPI(
    title="Invoice OCR API",
    description="AI-powered invoice processing with Gemini & Tesseract OCR",
    version="1.0.0",
    timeout=120  # 2 minute timeout for long processing
)

# Enable CORS for Postman and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS_DIR = os.path.join(os.getcwd(), "runs")
HISTORY_FILE = os.path.join(RUNS_DIR, "history.json")


@app.get('/')
def hello_json():
    """Return a simple JSON message for quick checks."""
    return {"message": "Hello world"}


@app.post('/ocr')
async def process_ocr(file: UploadFile = File(...)):
    """
    Upload an invoice PDF or image and extract structured JSON via OCR.
    Returns extracted invoice data as JSON.
    
    Note: Processing may take 10-30 seconds depending on file size and Gemini API.
    Increase timeout in your HTTP client if needed.
    """
    tmp_path = None
    print(f"\n{'='*70}")
    print(f"[API] New OCR request received")
    print(f"{'='*70}")
    try:
        print(f"\n[OCR] Processing file: {file.filename}")
        # Import processing modules
        from invoice_engine.barcode_extraction import extract_codes_from_images
        from invoice_engine.local_extraction import local_extract_invoice
        from invoice_engine.multipage_parser import parse_multipage_invoice
        
        # Save uploaded file to temp location
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
        print(f"[OCR] File type: {suffix}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Convert PDF to image(s) if needed
        page_bytes_list = None
        if suffix.lower() == ".pdf":
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(tmp_path)
                page_bytes_list = []
                for page in images:
                    buf = BytesIO()
                    page.save(buf, format="JPEG")
                    page_bytes_list.append(buf.getvalue())
                image_bytes = page_bytes_list[0] if page_bytes_list else b""
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"status": "ERROR", "error": "PDF conversion failed", "detail": str(e)}
                )
        else:
            with open(tmp_path, "rb") as imgf:
                image_bytes = imgf.read()
        
        # Run barcode/QR extraction
        try:
            if page_bytes_list:
                codes = extract_codes_from_images(page_bytes_list)
            else:
                codes = extract_codes_from_images([image_bytes])
        except Exception:
            codes = []
        
        # Try Gemini first (if API key present), then fallback to local OCR
        result_json = None
        prefer_gemini = bool(os.getenv("GEMINI_API_KEY"))
        print(f"[OCR] Gemini API key present: {prefer_gemini}")
        if prefer_gemini:
            try:
                print("[OCR] Attempting Gemini extraction...")
                from invoice_engine.vision_llm_gemini import extract_invoice_with_gemini
                gemini_images = page_bytes_list if page_bytes_list else [image_bytes]
                result_json = extract_invoice_with_gemini(gemini_images)
                print(f"[OCR] Gemini returned: {type(result_json).__name__}")
                # Check if Gemini returned NEEDS_REVIEW
                if isinstance(result_json, dict) and result_json.get("status") == "NEEDS_REVIEW":
                    print("[OCR] Gemini returned NEEDS_REVIEW, falling back to local OCR")
                    result_json = None
                else:
                    print("[OCR] ✓ Gemini extraction successful")
            except Exception as e:
                print(f"[OCR] Gemini failed with error: {e}")
                result_json = None
        
        # Fallback to local OCR if Gemini failed or not available
        if result_json is None:
            print("[OCR] Using local OCR extraction...")
            try:
                if page_bytes_list and len(page_bytes_list) > 1:
                    print(f"[OCR] Multi-page PDF detected ({len(page_bytes_list)} pages)")
                    result_json = parse_multipage_invoice(tmp_path)
                else:
                    print("[OCR] Single page/image extraction")
                    result_json = local_extract_invoice(image_bytes, lang="eng")
                print("[OCR] ✓ Local extraction successful")
            except Exception as e:
                print(f"[OCR] Local extraction failed: {e}")
                result_json = {"status": "NEEDS_REVIEW", "error": str(e), "codes": codes}
        
        # Normalize result_json
        if isinstance(result_json, str):
            try:
                result_json = json.loads(result_json)
            except Exception:
                result_json = {"status": "NEEDS_REVIEW", "raw_text": result_json}
        
        if isinstance(result_json, dict):
            result_json["codes"] = codes
        else:
            result_json = {
                "status": "NEEDS_REVIEW",
                "error": "Invalid extractor response",
                "raw_response": str(result_json),
                "codes": codes,
            }
        
        return JSONResponse(content=result_json)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "error": "Processing failed", "detail": str(e)}
        )
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.get("/health")
def health():
    runs_writable = os.path.isdir(RUNS_DIR) and os.access(RUNS_DIR, os.W_OK)
    return {"status": "ok" if runs_writable else "degraded", "runs_dir_writable": runs_writable, "timestamp": datetime.utcnow().isoformat() + "Z"}


if __name__ == "__main__":
    try:
        import uvicorn
        # Bind to 0.0.0.0 to allow access from other IP addresses
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception:
        print("Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload")
