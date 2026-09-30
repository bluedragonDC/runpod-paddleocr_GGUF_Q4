FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OCR_CACHE_DIR=/app/models
ENV LLAMA_SERVER=/app/llama.cpp/llama-server

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip git build-essential cmake \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && ln -sf /usr/bin/python3.11 /usr/bin/python

# Build llama.cpp with strict CUDA support
RUN mkdir -p /app/models \
    && git clone https://github.com/ggerganov/llama.cpp.git /app/llama.cpp \
    && cd /app/llama.cpp \
    && cmake -B build -DGGML_CUDA=ON \
    && cmake --build build --config Release -j 8 \
    && cp build/bin/llama-server /app/llama.cpp/

# Install python packages
RUN pip install --no-cache-dir runpod huggingface-hub pymupdf requests

# Pre-download models
RUN python3 -c "\
from huggingface_hub import hf_hub_download;\
hf_hub_download('mradermacher/PaddleOCR-VL-1.6-GGUF', 'PaddleOCR-VL-1.6.Q4_K_M.gguf', local_dir='/app/models');\
hf_hub_download('PaddlePaddle/PaddleOCR-VL-1.6-GGUF', 'PaddleOCR-VL-1.6-GGUF-mmproj.gguf', local_dir='/app/models');\
"

WORKDIR /app
COPY . /app

CMD ["python3", "-u", "handler.py"]
