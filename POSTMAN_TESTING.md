# Testing Invoice OCR API in Postman

## 🚀 Quick Start

### Step 1: Start the Server Locally

```bash
uvicorn api_server:app --reload
```

Server will run at: `http://127.0.0.1:8000`

---

## 📡 API Endpoints for Postman

### 1️⃣ Health Check (GET)

**Method:** `GET`  
**URL:** `http://127.0.0.1:8000/health`  
**Headers:** None required

**Expected Response:**
```json
{
  "status": "ok",
  "runs_dir_writable": true,
  "timestamp": "2026-02-02T10:30:45.123456Z"
}
```

---

### 2️⃣ Hello World (GET)

**Method:** `GET`  
**URL:** `http://127.0.0.1:8000/`  
**Headers:** None required

**Expected Response:**
```json
{
  "message": "Hello world"
}
```

---

### 3️⃣ OCR Processing (POST) - MAIN ENDPOINT

**Method:** `POST`  
**URL:** `http://127.0.0.1:8000/ocr`

#### Postman Setup:

1. **Select POST method**
2. **Enter URL:** `http://127.0.0.1:8000/ocr`
3. **Go to "Body" tab**
4. **Select "form-data"**
5. **Add a new key:**
   - **Key:** `file` (change type to "File" using dropdown)
   - **Value:** Click "Select Files" and choose your PDF/image

#### Screenshot Guide:
```
┌─────────────────────────────────────────────┐
│ POST  http://127.0.0.1:8000/ocr       Send │
├─────────────────────────────────────────────┤
│ Params   Authorization   Headers   Body    │
├─────────────────────────────────────────────┤
│ ○ none                                      │
│ ○ form-data  ✓ (selected)                  │
│ ○ x-www-form-urlencoded                     │
│ ○ raw                                       │
│ ○ binary                                    │
│                                             │
│ KEY          VALUE                 TYPE     │
│ file         [Select Files]        File ▼  │
└─────────────────────────────────────────────┘
```

#### Expected Response (Success):
```json
{
  "status": "EXTRACTED",
  "document_type": "invoice",
  "header": {
    "vendor_details": {
      "company_name_en": "ABC Company",
      "tax_registration_number": "123456789",
      ...
    },
    "invoice_details": {
      "invoice_number": "INV-001",
      "invoice_date": "2026-02-01",
      ...
    }
  },
  "line_items": [
    {
      "prod_code": "PROD-001",
      "product_name": "Widget",
      "qty": 10,
      "unit_price": 50.00,
      "amount": 500.00
    }
  ],
  "summary": {
    "subtotal": 500.00,
    "vat_total": 75.00,
    "total_amount": 575.00
  },
  "codes": ["BARCODE123"],
  "_debug": {
    "_gemini_diagnostics": {
      "parsed_json_keys": [...]
    }
  }
}
```

#### Expected Response (Error):
```json
{
  "status": "ERROR",
  "error": "Processing failed",
  "detail": "Error message here"
}
```

---

## 🧪 Testing with Different Files

### Test Files to Try:

1. **PDF Invoice** - `runs/20260202T044111Z_7e0e6d89_1004 gormet.pdf`
2. **Image Invoice** - Any `.jpg` or `.png` invoice
3. **Multi-page PDF** - PDF with multiple invoice pages

---

## 🔧 Advanced Postman Features

### Save as Collection:

1. Click "Save" after setting up the request
2. Create a new collection: "Invoice OCR API"
3. Save all three endpoints (GET /, GET /health, POST /ocr)

### Environment Variables:

Create a Postman environment:

| Variable | Initial Value | Current Value |
|----------|---------------|---------------|
| base_url | http://127.0.0.1:8000 | http://127.0.0.1:8000 |
| deployed_url | https://your-app.onrender.com | https://your-app.onrender.com |

Then use: `{{base_url}}/ocr`

---

## 📊 Testing Checklist

- [ ] GET `/health` returns `200 OK`
- [ ] GET `/` returns `{"message": "Hello world"}`
- [ ] POST `/ocr` with PDF returns structured JSON
- [ ] POST `/ocr` with image returns structured JSON
- [ ] Response includes `codes` array (barcodes)
- [ ] Response includes `_debug._gemini_diagnostics` (if Gemini used)
- [ ] Response time < 30 seconds

---

## 🌐 Testing Deployed API

Once deployed to Render, change the URL to:

```
https://your-app-name.onrender.com/ocr
```

**Note:** First request after inactivity may take 30-60 seconds (free tier spin-up)

---

## 📸 Example Postman Request (cURL Export)

```bash
curl --location 'http://127.0.0.1:8000/ocr' \
--form 'file=@"/path/to/invoice.pdf"'
```

---

## ✨ Interactive API Docs (Alternative to Postman)

FastAPI provides built-in interactive docs:

**Open in browser:** `http://127.0.0.1:8000/docs`

This gives you a Swagger UI where you can:
- See all endpoints
- Test requests directly in browser
- See request/response schemas
- Download OpenAPI spec

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Start server with `uvicorn api_server:app --reload` |
| 500 Error | Check server logs for error details |
| Slow response | Gemini API calls take 10-30 seconds |
| No Gemini diagnostics | Check GEMINI_API_KEY is set in environment |

---

## 📱 Postman Collection Export (JSON)

Save this as `Invoice_OCR_API.postman_collection.json`:

```json
{
  "info": {
    "name": "Invoice OCR API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "header": [],
        "url": {
          "raw": "{{base_url}}/health",
          "host": ["{{base_url}}"],
          "path": ["health"]
        }
      }
    },
    {
      "name": "Hello World",
      "request": {
        "method": "GET",
        "header": [],
        "url": {
          "raw": "{{base_url}}/",
          "host": ["{{base_url}}"],
          "path": [""]
        }
      }
    },
    {
      "name": "OCR Process Invoice",
      "request": {
        "method": "POST",
        "header": [],
        "body": {
          "mode": "formdata",
          "formdata": [
            {
              "key": "file",
              "type": "file",
              "src": []
            }
          ]
        },
        "url": {
          "raw": "{{base_url}}/ocr",
          "host": ["{{base_url}}"],
          "path": ["ocr"]
        }
      }
    }
  ],
  "variable": [
    {
      "key": "base_url",
      "value": "http://127.0.0.1:8000"
    }
  ]
}
```

**Import this file into Postman:** File → Import → Select file

---

**Happy Testing! 🚀**
