# omniASR Streaming Server
# Supports: CUDA (NVIDIA GPU), CPU
# Note: MPS (Apple Silicon) not available in Docker

# ============================================
# Stage 1: Builder
# ============================================
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    libsndfile1 \
    git \
    curl \
    build-essential \
    cmake \
    ca-certificates

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Create virtual environment
RUN python3 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Install dependencies
COPY requirements.txt .
COPY ten-vad ./ten-vad
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt && \
    uv pip install ./ten-vad

# ============================================
# Stage 2: Runtime
# ============================================
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libsndfile1 \
    ffmpeg \
    curl

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# ============================================
# Models (Cached Layer)
# ============================================
# Copy models cache BEFORE app code so code changes don't invalidate the layer
ENV FAIRSEQ2_CACHE_DIR=/models/fairseq2/assets
COPY models-cache /models
RUN chmod -R a+rX /models

# ============================================
# Application code
# ============================================
COPY *.py .
COPY *.html .

# ============================================
# Configuration
# ============================================
# Server settings
ENV HOST=0.0.0.0
ENV PORT=8080

# Environment variables from .env
ENV MAX_CONCURRENT_REQUESTS=100
ENV MAX_WEBSOCKET_CONNECTIONS=50
ENV CHUNK_DURATION=3.0
ENV VAD_ENABLED=true

# Model settings
ENV MODEL_CARD=omniASR_LLM_1B_v2
ENV DEFAULT_LANG=
ENV DEVICE=

# Expose port
EXPOSE ${PORT}

# Run the server
CMD ["python", "server.py"]
