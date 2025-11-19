# Base dari pre-built llama-server dengan CUDA support (ringan, ~1.5GB)
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# Set non-interaktif mode untuk apt dan timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Jakarta

# Set workdir
WORKDIR /app

# Instal Python 3.12 dan deps sistem + DeadSnakes PPA
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/* && \
    python3.12 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

# Instal app deps dari requirements.txt
COPY requirements.txt .
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy codebase kamu (app, config, dll.)
COPY . .

# Copy json config ke direktori yang sesuai
COPY config.json .

# Download models ke /app/models selama build (sequential biar nggak overload)
RUN mkdir -p models
RUN wget -P models https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_1.gguf
RUN wget -P models https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-f16.gguf

# Set ENV untuk llama-server path
ENV LLAMA_SERVER_PATH=/app/llama-server

# Untuk logging real-time
ENV PYTHONUNBUFFERED=1

# Expose port API
EXPOSE 8000

# ENTRYPOINT set ke venv
ENTRYPOINT ["/venv/bin/python"]

# Command run (jalankan run.py)
CMD ["run.py"]