# 📄 Universal Invoice Processing Engine

### AI-Powered PDF & Image → Structured JSON Converter

> A robust, production-ready system that converts **multi-page invoices (PDF/Image)** into a **clean, validated JSON schema** using OCR and modern AI models.

---

## ✨ Key Features

* 📑 **Multi-page PDF & Image support**
* 🔍 **High-accuracy OCR** (Tesseract + PDF parsing)
* 🧠 **AI-assisted JSON extraction** (Gemini / LLM-based)
* 🧾 **Strict invoice JSON schema validation**
* 🧠 **Handles messy, real-world invoices**
* 🧩 **Modular architecture (ERP-ready)**
* 🌐 **Web UI powered by Streamlit**
* 🌐 **REST API powered by FastAPI**
* ☁️ **Cloud deployable (Render / Railway / VPS)**

---

## 🏗️ Architecture Overview

```
image_pdf_to_json/
│
├── streamlit_app.py            # Web UI
├── api_server.py               # FastAPI REST API
│
├── invoice_engine/
│   ├── local_extraction.py     # OCR & text extraction
│   ├── multipage_parser.py     # Multi-page invoice logic
│   ├── barcode_extraction.py   # Barcode / QR (optional)
│   ├── vision_llm_gemini.py    # Gemini AI extraction
│   ├── universal_schema.py     # Invoice JSON schema
│   └── orchestrator.py         # Processing orchestration
│
├── requirements.txt
├── runtime.txt                 # Python version (3.11)
├── Dockerfile                  # Docker configuration
├── render.yaml                 # Render deployment config
├── DEPLOYMENT.md               # Deployment guide
└── README.md
```

---

## 📂 Supported Inputs

* ✅ PDF (single & multi-page)
* ✅ Scanned invoices
* ✅ Camera images
* ✅ Mixed text + image invoices

---

## 🧾 Output Format

The system produces a **structured JSON** including:

* Vendor details
* Invoice metadata
* Customer information
* Line items
* Taxes & totals
* Payment instructions
* Barcode/QR codes
* Validation confidence

> Designed to plug directly into **ERP / Accounting systems**

---

## ⚙️ Tech Stack

| Layer            | Technology            |
| ---------------- | --------------------- |
| UI               | Streamlit             |
| API              | FastAPI + Uvicorn     |
| OCR              | Tesseract, PDFPlumber |
| Image Processing | OpenCV                |
| AI / LLM         | Gemini API            |
| Validation       | Custom JSON schema    |
| Deployment       | Render / Railway      |
| Language         | Python 3.11           |

---

## 🚀 Getting Started (Local Setup)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/image_pdf_to_json.git
cd image_pdf_to_json
```

### 2️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ On Windows, install **Poppler** and **Tesseract** separately for PDF/OCR support.

---

### 4️⃣ Set environment variables

Create a `.env` file:

```bash
GEMINI_API_KEY=your_api_key_here
```

(Use Render / Railway dashboard for production)

---

### 5️⃣ Run the app

**Option A: Web UI (Streamlit)**
```bash
streamlit run streamlit_app.py
```

**Option B: REST API (FastAPI)**
```bash
uvicorn api_server:app --reload
```

Then visit: http://127.0.0.1:8000/docs for interactive API documentation

---

## 📡 API Endpoints

### `POST /ocr`
Upload and process an invoice

**Request:**
```bash
curl -X POST http://localhost:8000/ocr \
  -F "file=@invoice.pdf"
```

**Response:**
```json
{
  "status": "EXTRACTED",
  "document_type": "invoice",
  "company": {...},
  "line_items": [...],
  "summary": {...},
  "codes": ["barcode1"],
  "_debug": {...}
}
```

### `GET /health`
Check server status

### `GET /`
Hello world test endpoint

---

## ☁️ Deployment (Render)

This project includes a **ready-to-use `render.yaml`**.

System dependencies installed automatically:

* `libzbar0`
* `libgl1`
* `tesseract-ocr`
* `poppler-utils`

Deploy steps:

1. Push code to GitHub
2. Create a new Render Web Service
3. Select repository
4. Add `GEMINI_API_KEY` in Environment Variables
5. Click **Deploy**

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

---

## 🧠 Design Decisions

* **LLM used only for intelligence**, not raw OCR
* **Defensive imports** for optional native dependencies
* Barcode detection is **optional**, not blocking
* Built for **real-world invoice noise**
* **Dual interface**: Web UI + REST API for maximum flexibility

---

## 🔐 Security Notes

* ❌ No API keys committed to repo
* ✅ Environment-based secrets
* ✅ `.env` file is git-ignored
* ✅ Safe for production & demos

---

## 🧪 Testing

**Test the API locally:**
```bash
python test_with_gemini.py
```

**Test deployed API:**
```bash
# Update RENDER_URL in test_deployed_api.py first
python test_deployed_api.py
```

---

## 📈 Future Enhancements

* 🔄 Async batch processing
* 🧠 Auto-confidence scoring
* 🧾 Line-item reconciliation logic
* 📊 ERP / SAP / Tally integrations
* 🔍 Table structure detection
* 🔐 Authentication & rate limiting

---

## 📄 License

MIT License - Feel free to use in commercial projects

---

## 👨‍💻 Author

**@Sri174 - VEERACHINNU M**

---

## 🙏 Acknowledgments

* Google Gemini API for AI-powered extraction
* Tesseract OCR for text recognition
* FastAPI for modern API framework
* Streamlit for rapid UI development

---

**⭐ Star this repo if it helped you!**
