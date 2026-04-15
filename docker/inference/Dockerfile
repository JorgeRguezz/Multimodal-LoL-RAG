FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
ENV SANITIZED_CACHE_ROOT=/app/runtime_data/sanitized_cache
ENV SANITIZED_BUILD_GLOB=sanitized_build_cache_*
ENV SANITIZED_GLOBAL_ROOT=/app/runtime_data/sanitized_cache/sanitized_global
ENV SANITIZED_GLOBAL_GRAPH=/app/runtime_data/sanitized_cache/sanitized_global/graph_AetherNexus.graphml
ENV SANITIZED_GLOBAL_MANIFEST=/app/runtime_data/sanitized_cache/sanitized_global/aether_manifest.json
ENV VIDEO_METADATA_REGISTRY=/app/knowledge_inference/video_metadata.json
ENV OSS_LOCAL_DIR=/app/models/gpt-oss-20b

# For llama-cpp-python CUDA support
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
ENV PIP_NO_BINARY=llama-cpp-python

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    git \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Install Node.js (for the frontend) (compatible with Debian-based images)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip

RUN pip install -r knowledge_pipeline/deploy/requirements-smolvlm.txt

RUN pip install --no-cache-dir llama-cpp-python==0.3.16

RUN cd /app/knowledge_frontend && npm ci && npm run build

CMD ["bash", "-lc", "uvicorn knowledge_api_server.main:app --host 0.0.0.0 --port 8000 & cd /app/knowledge_frontend && npm run start -- --hostname 0.0.0.0 --port 3000"]
