FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OCR_CACHE_DIR=/app/models
ENV LLAMA_SERVER=/usr/local/bin/llama-server

# Install minimal dependencies (NO cmake, NO build-essential needed!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip python3.11 python3.11-venv python3-pip libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && ln -sf /usr/bin/python3.11 /usr/bin/python

# Download the pre-built CUDA Wheel for llama-cpp-python (which includes llama-server binary)
# This takes 10 seconds instead of 45 minutes!
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir llama-cpp-python[server] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Find the installed llama-server binary and symlink it to our expected path
RUN ln -sf $(which python3) /usr/local/bin/llama-server

# Install Python packages
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
