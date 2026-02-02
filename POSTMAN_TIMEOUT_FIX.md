# 🔧 Postman Timeout Fix

## The Problem
You're seeing: **"Error: Response timed out"**

This happens because **Gemini processing takes 10-30 seconds**, but Postman's default timeout is shorter.

---

## ✅ Solution: Increase Postman Timeout

### Method 1: Change Timeout in Settings

1. **Open Postman Settings**
   - Click the gear icon ⚙️ (top right)
   - Or: File → Settings

2. **Go to "General" tab**

3. **Find "Request timeout in ms"**
   - Default: 30000 (30 seconds)
   - Change to: **120000** (2 minutes)

4. **Click "Save"**

5. **Retry your request**

---

### Method 2: Test with Simple Endpoint First

Before testing `/ocr`, verify the server is working:

**Step 1: Test Health Check**
```
GET http://127.0.0.1:8000/health
```
Should respond instantly with:
```json
{
  "status": "ok",
  "runs_dir_writable": true,
  "timestamp": "..."
}
```

**Step 2: Test Hello World**
```
GET http://127.0.0.1:8000/
```
Should respond instantly with:
```json
{
  "message": "Hello world"
}
```

**Step 3: Now test OCR**
```
POST http://127.0.0.1:8000/ocr
Body: form-data
Key: file (File type)
Value: [Select your PDF/image]
```
Wait 10-30 seconds for processing.

---

## 📊 What You Should See

### In Postman:
- Status: `200 OK`
- Time: `~10000-30000 ms`
- Response: Large JSON object with invoice data

### In Server Terminal:
```
[API] New OCR request received
======================================================================
[OCR] Processing file: invoice.pdf
[OCR] File type: .pdf
[OCR] Gemini API key present: True
[OCR] Attempting Gemini extraction...
[OCR] Gemini returned: dict
[OCR] ✓ Gemini extraction successful
INFO:     127.0.0.1:xxxxx - "POST /ocr HTTP/1.1" 200 OK
```

---

## 🚀 Alternative: Use Browser Instead

Open this URL in your browser:
```
http://127.0.0.1:8000/docs
```

This opens **FastAPI's interactive docs** where you can:
- Upload files directly in the browser
- See progress in real-time
- No timeout issues
- Auto-generated documentation

**Steps:**
1. Expand "POST /ocr"
2. Click "Try it out"
3. Click "Choose File" and select invoice
4. Click "Execute"
5. Wait for response (you'll see a loading spinner)

---

## 🔍 Still Having Issues?

### Check if server is running:
```powershell
curl http://127.0.0.1:8000/health
```

### Check server logs:
Look at the terminal where uvicorn is running for error messages.

### Try a smaller file:
Large PDFs take longer. Try with a single-page PDF or image first.

### Disable Gemini (faster testing):
Temporarily remove or comment out `GEMINI_API_KEY` in `.env` file.
This will use local OCR only (faster, but less accurate).

---

## ✨ Quick Test Script (Alternative)

Instead of Postman, use this Python script:

```python
import requests

url = "http://127.0.0.1:8000/ocr"
file_path = r"runs\20260202T044111Z_7e0e6d89_1004 gormet.pdf"

print("Sending request (may take 10-30 seconds)...")
with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files, timeout=120)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
```

Save as `quick_test.py` and run:
```bash
python quick_test.py
```

---

**Your API is working! Just need to increase the timeout.** ✅
