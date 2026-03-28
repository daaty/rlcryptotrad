# ─────────────────────────────────────────────────────────────
# Trading Bot Dashboard — Docker image
# Base: python:3.11-slim  (no CUDA; CPU inference)
# Build: docker build -t trading-bot .
# Run:   docker compose up
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps for TA-Lib, numpy, pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Build & install TA-Lib C library (required by TA-Lib python binding)
RUN wget -q https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr && make -j$(nproc) && make install \
    && cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# Copy source code
COPY . .

# Create runtime directories
RUN mkdir -p data logs/trading models

# Streamlit configuration — disable browser auto-open, bind on all interfaces
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

# Health-check: wait for Streamlit to respond
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD wget -qO- http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard_new.py", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.address=0.0.0.0"]
