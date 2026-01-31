FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless true
