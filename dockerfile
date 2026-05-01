# CUDA 12.1 + cuDNN 8 on Ubuntu 22.04.
# vllm 0.19+ requires CUDA 12.x; PyTorch cu118 wheels still install cleanly
# on CUDA 12 drivers.  Adjust the tag if your cluster runs a different toolkit.
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Put uv and the venv on PATH
    PATH="/root/.local/bin:/app/.venv/bin:$PATH"

# ── System packages ────────────────────────────────────────────────────────────
# git     – required at install time for the swagger-client GitHub dependency
# python-is-python3 / python3-venv – uv needs a base interpreter
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        git \
        ca-certificates \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
    && rm -rf /var/lib/apt/lists/*

# ── uv ─────────────────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# ── Install dependencies (cached layer) ────────────────────────────────────────
# Copy only the lock + manifest first so this layer is not invalidated by
# source-code changes.
COPY pyproject.toml uv.lock ./

RUN uv sync \
        --python 3.12 \
        --frozen \
        --no-install-project \
        --all-groups

# ── Install the project itself ──────────────────────────────────────────────────
COPY align_system/ ./align_system/
COPY README.md README.md

RUN uv sync \
        --python 3.12 \
        --frozen \
        --all-groups \
        --all-packages

# ── Default entrypoint ─────────────────────────────────────────────────────────
ENTRYPOINT ["uv", "run", "run_align_system"]
