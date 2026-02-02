FROM python:3.11-slim

# System dependencies for OCR, image processing, and barcode reading
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libzbar0 \
    libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create runs directory for output files
RUN mkdir -p runs

EXPOSE 8000

# Run FastAPI server with uvicorn
CMD uvicorn api_server:app --host 0.0.0.0 --port $PORT
