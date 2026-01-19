# HybridRAG Docker Image
# State-of-the-art RAG with MongoDB Atlas + Voyage AI
#
# Build:
#   docker build -t hybridrag:latest .
#
# Run API:
#   docker run -p 8000:8000 --env-file .env hybridrag:latest
#
# Run UI:
#   docker run -p 8001:8001 --env-file .env hybridrag:latest chainlit run src/hybridrag/ui/chat.py --port 8001 --host 0.0.0.0
#

# Build stage
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY pyproject.toml .
COPY src/ src/

# Create virtual environment and install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install with API extras (most common deployment)
RUN pip install --upgrade pip && \
    pip install -e ".[api,cli]"


# Production stage
FROM python:3.12-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src"

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY src/ src/
COPY pyproject.toml .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Default port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command - start FastAPI server
CMD ["uvicorn", "hybridrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
