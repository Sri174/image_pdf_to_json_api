"""
streamlit_app.py
----------------
Streamlit GUI for the invoice processing engine.
 Upload PDF/image
- Show extracted JSON, validation, duplicate check, and SAP payload
"""

import streamlit as st
import logging
import os
# Reduce noisy Streamlit runtime warnings when the app is executed directly with `python`.
# Recommended: run the app with the Streamlit CLI: `streamlit run streamlit_app.py`
logging.getLogger("streamlit").setLevel(logging.WARNING)

# Load .env file into environment (simple parser) so Streamlit sees API keys when running via Python
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
                # don't override existing env vars
                if k and not os.getenv(k):
                    os.environ[k] = v
    except Exception:
        pass
from PIL import Image
import tempfile
import os
import json
import shutil
import copy
from datetime import datetime
import uuid
# Removed Vision LLM (Gemini) integration: use local OCR/multipage parser instead
from invoice_engine.barcode_extraction import extract_codes_from_images
from invoice_engine.local_extraction import local_extract_invoice
from invoice_engine.multipage_parser import parse_multipage_invoice
from invoice_engine.duplicate_check import is_duplicate
from invoice_engine.pages_to_json import consolidate_invoice_from_pages
import pytesseract

EXISTING_INVOICES = []  # In production, load from DB or file

st.title("Universal Invoice Processing Engine (DONUT + SAP)")

# Ensure runs directory exists for history
RUNS_DIR = os.path.join(os.getcwd(), "runs")
os.makedirs(RUNS_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(RUNS_DIR, "history.json")
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as hf:
        json.dump([], hf)

uploaded_file = st.file_uploader("Upload Invoice (PDF or Image)", type=["pdf", "jpg", "jpeg", "png", "bmp"])

# Top placeholder for download/info (will be filled or updated after extraction)
top_placeholder = st.empty()
if st.session_state.get("last_result_json"):
    try:
        top_name = st.session_state.get("last_uploaded_name", "extracted_invoice.json")
        top_dl_name = os.path.splitext(top_name)[0] + ".json"
    except Exception:
        top_dl_name = "extracted_invoice.json"
    top_placeholder.download_button(
        label="Download Last Extracted JSON",
        data=json.dumps(st.session_state["last_result_json"], indent=2).encode("utf-8"),
        file_name=top_dl_name,
        mime="application/json",
        key="download_top",
    )
else:
    top_placeholder.info("No extracted invoice JSON available yet — upload a file to run extraction.")

tab1, tab2 = st.tabs(["Process", "History"])

with tab1:
    # Automatic selection: prefer remote Mistral API when `MISTRAL_API_KEY` is present, otherwise use local OCR heuristics.
    # OCR language selection for pytesseract when using local-first
    lang_map = {
        "Auto (use Tesseract default)": None,
        "English": "eng",
        "Spanish": "spa",
        "German": "deu",
        "French": "fra",
        "Italian": "ita",
    }
    if "ocr_lang" not in st.session_state:
        st.session_state["ocr_lang"] = "eng"
    selected_label = st.selectbox("OCR language (local-first)", list(lang_map.keys()), index=1)
    selected_lang = lang_map.get(selected_label)
    st.session_state["ocr_lang"] = selected_lang
    if not uploaded_file:
        st.info("Please upload an invoice PDF or image.")
    else:
        # Show uploaded file info for debugging
        try:
            st.write(f"Uploaded: {uploaded_file.name} ({uploaded_file.type}, {uploaded_file.size} bytes)")
        except Exception:
            pass

        tmp_path = None
        try:
            # Save to temp file
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            st.info(f"Processing: {uploaded_file.name}")

            # Show top-area processing info
            try:
                top_placeholder.info("Processing invoice — extracting barcodes and running local OCR...")
            except Exception:
                pass

            # Run extraction under a visible spinner
            with st.spinner("Extracting barcodes and running local OCR..."):
                # Convert PDF to image(s) if needed
                page_bytes_list = None
                if suffix.lower() == ".pdf":
                    try:
                        from pdf2image import convert_from_path
                        from io import BytesIO
                        images = convert_from_path(tmp_path)
                        page_bytes_list = []
                        for page in images:
                            buf = BytesIO()
                            page.save(buf, format="JPEG")
                            page_bytes_list.append(buf.getvalue())
                        # Use first page when needed for local heuristics
                        image_bytes = page_bytes_list[0] if page_bytes_list else b""
                    except Exception as e:
                        st.error("Failed to convert PDF to images. Ensure Poppler is installed and the path is correct.")
                        st.exception(e)
                        image_bytes = b""
                else:
                    with open(tmp_path, "rb") as imgf:
                        image_bytes = imgf.read()

                # Run barcode/QR extraction first (deterministic)
                if page_bytes_list:
                    codes = extract_codes_from_images(page_bytes_list)
                else:
                    codes = extract_codes_from_images([image_bytes])

                # Always prefer Gemini: send image(s) directly to Gemini for JSON conversion
                result_json = None
                prefer_gemini = bool(os.getenv("GEMINI_API_KEY"))
                if prefer_gemini:
                    try:
                        from invoice_engine.vision_llm_gemini import extract_invoice_with_gemini

                        try:
                            gemini_images = page_bytes_list if page_bytes_list else [image_bytes]
                            result_json = extract_invoice_with_gemini(gemini_images)
                        except Exception:
                            result_json = None
                    except Exception:
                        result_json = None
                # If Gemini returned a NEEDS_REVIEW dict, surface diagnostics to the UI so user sees why it failed
                try:
                    if isinstance(result_json, dict) and result_json.get("status") == "NEEDS_REVIEW":
                        st.warning("Gemini returned a NEEDS_REVIEW response — falling back to local extraction.")
                        try:
                            st.json(result_json)
                        except Exception:
                            st.write(result_json)
                        # continue to fallback below
                        result_json = None
                except Exception:
                    pass

                if result_json is None:
                    try:
                        if page_bytes_list and len(page_bytes_list) > 1:
                            result_json = parse_multipage_invoice(tmp_path)
                        else:
                            result_json = local_extract_invoice(image_bytes, lang=st.session_state.get("ocr_lang"))
                    except Exception as e:
                        st.error("Local OCR extraction failed")
                        st.exception(e)
                        result_json = {"status": "NEEDS_REVIEW", "error": str(e), "codes": codes}

                # Normalize result_json to a dict so Streamlit shows structured JSON
                try:
                    if isinstance(result_json, str):
                        # try to parse if it's a JSON string
                        try:
                            result_json = json.loads(result_json)
                        except Exception:
                            result_json = {"status": "NEEDS_REVIEW", "raw_text": result_json}
                    if isinstance(result_json, dict):
                        # If extractor returned only raw OCR text in _debug, try lightweight heuristic conversion
                        try:
                            dbg = result_json.get("_debug", {})
                            raw_text = dbg.get("raw_text")
                            li = result_json.get("line_items") or []
                            if raw_text and (not li or len(li) == 0):
                                try:
                                    from invoice_engine.local_extraction import parse_raw_text_to_json
                                    conv = parse_raw_text_to_json(raw_text)
                                    # preserve any detected codes
                                    conv["codes"] = result_json.get("codes", [])
                                    result_json = conv
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        result_json["codes"] = codes
                    else:
                        # fallback wrapper
                        result_json = {
                            "status": "NEEDS_REVIEW",
                            "error": "Invalid extractor response",
                            "raw_response": str(result_json),
                            "codes": codes,
                        }
                except Exception:
                    result_json = {"status": "NEEDS_REVIEW", "error": "normalization_failed", "raw_response": str(result_json), "codes": codes}

            st.subheader("Extracted Invoice JSON (Local OCR)")
            st.json(result_json)

            # Consolidation action removed: consolidated OCR-pages conversion
            # was previously triggered by a button here. Removed per user request.

            # (Manual crop OCR removed — using API/automated extraction instead)

            # Persist run: copy original file and save extracted JSON + metadata
            run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
            try:
                original_name = f"{run_id}_{uploaded_file.name}"
                saved_original = os.path.join(RUNS_DIR, original_name)
                shutil.copy(tmp_path, saved_original)
            except Exception:
                saved_original = None

            try:
                json_name = f"{run_id}_extracted.json"
                saved_json = os.path.join(RUNS_DIR, json_name)
                with open(saved_json, "w", encoding="utf-8") as jf:
                    json.dump(result_json, jf, ensure_ascii=False, indent=2)
            except Exception:
                saved_json = None

            # Append to history index
            try:
                with open(HISTORY_FILE, "r+", encoding="utf-8") as hf:
                    history = json.load(hf)
                    entry = {
                        "run_id": run_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "original_file": os.path.basename(saved_original) if saved_original else None,
                        "extracted_json": os.path.basename(saved_json) if saved_json else None,
                        "status": result_json.get("status", "NEEDS_REVIEW")
                    }
                    history.insert(0, entry)
                    hf.seek(0)
                    json.dump(history, hf, ensure_ascii=False, indent=2)
                    hf.truncate()
            except Exception:
                pass

            # Save to session state so the top download button can access it
            st.session_state["last_result_json"] = result_json
            try:
                st.session_state["last_uploaded_name"] = uploaded_file.name
            except Exception:
                st.session_state["last_uploaded_name"] = "extracted_invoice.json"

            # Update top placeholder with immediate download button
            try:
                top_dl_name = os.path.basename(saved_json) if saved_json else "extracted_invoice.json"
                top_placeholder.download_button(
                    label="Download Last Extracted JSON",
                    data=json.dumps(result_json, indent=2).encode("utf-8"),
                    file_name=top_dl_name,
                    mime="application/json",
                    key="download_top_after"
                )
            except Exception:
                pass

            # Allow downloading original file and extracted JSON locally from the UI
            with st.expander("Download files for this execution"):
                if saved_json and os.path.exists(saved_json):
                    with open(saved_json, "rb") as f:
                        st.download_button("Download extracted JSON", f.read(), file_name=os.path.basename(saved_json), mime="application/json")
                if saved_original and os.path.exists(saved_original):
                    with open(saved_original, "rb") as f:
                        st.download_button("Download original uploaded file", f.read(), file_name=os.path.basename(saved_original), mime="application/octet-stream")

            # Duplicate check
            duplicate = is_duplicate(result_json, EXISTING_INVOICES)
            st.subheader("Duplicate Check")
            st.write(duplicate)

        except Exception as e:
            st.error("Unexpected error during processing")
            st.exception(e)
        finally:
            # Cleanup temp file
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

with tab2:
    st.header("Extraction History")
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as hf:
            history = json.load(hf)
    except Exception:
        history = []
    if not history:
        st.info("No previous runs found.")
    else:
        for entry in history:
            cols = st.columns([2, 1, 1, 1])
            with cols[0]:
                st.write(f"**{entry.get('run_id')}** — {entry.get('timestamp')}")
                st.write(f"Original: {entry.get('original_file')}, Extracted: {entry.get('extracted_json')}")
            with cols[1]:
                st.write(entry.get('status'))
            with cols[2]:
                if entry.get('extracted_json'):
                    path = os.path.join(RUNS_DIR, entry.get('extracted_json'))
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            st.download_button("Download JSON", f.read(), file_name=entry.get('extracted_json'), mime="application/json")
            with cols[3]:
                if entry.get('original_file'):
                    path = os.path.join(RUNS_DIR, entry.get('original_file'))
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            st.download_button("Download Original", f.read(), file_name=entry.get('original_file'), mime="application/octet-stream")
            st.markdown("---")
