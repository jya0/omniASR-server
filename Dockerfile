# omniASR Streaming Server
# Uses base image with models, installs dependencies here

FROM localhost/adeo-omniasr-base:latest

ENV DEBIAN_FRONTEND=noninteractive

# ============================================
# System dependencies
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    build-essential \
    cmake \
    ca-certificates \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust/Cargo (required for deepfilterlib)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Create virtual environment using uv (faster)
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# ============================================
# Python dependencies (using lock file from working venv)
# ============================================
COPY requirements-lock.txt .
COPY ten-vad ./ten-vad
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -r requirements-lock.txt && \
    uv pip install ./ten-vad

# Install LLVM C++ runtime required by ten_vad's native library
RUN apt-get update && apt-get install -y --no-install-recommends \
    libc++1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Application code
# ============================================
COPY *.py .

# ============================================
# Model cache paths (must match Dockerfile.base)
# ============================================
ENV FAIRSEQ2_CACHE_DIR=/models/fairseq2/assets
ENV XDG_CACHE_HOME=/models

# ============================================
# Configuration
# ============================================
ENV HOST=0.0.0.0
ENV PORT=8080
ENV MAX_CONCURRENT_REQUESTS=100
ENV MAX_WEBSOCKET_CONNECTIONS=50
ENV VAD_ENABLED=true
ENV NOISE_REMOVAL_ENABLED=true
ENV MODEL_CARD=omniASR_LLM_1B_v2
ENV DEFAULT_LANG=
ENV DEVICE=

# ============================================
# Redirect all writes to /tmp for read-only filesystem compatibility
# ============================================
ENV HOME=/tmp
ENV TMPDIR=/tmp
ENV TEMP=/tmp
ENV TMP=/tmp
# Python/pip cache
ENV PIP_CACHE_DIR=/tmp/.cache/pip
ENV PYTHONPYCACHEPREFIX=/tmp/__pycache__
# DeepFilterNet logs go to current working dir or HOME
ENV DF_LOG_DIR=/tmp

# Ensure /tmp is writable by any UID (OpenShift/K8s)
RUN chmod 1777 /tmp

EXPOSE ${PORT}

CMD ["python", "server.py"]
