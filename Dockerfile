FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OCR_CACHE_DIR=/app/models
ENV LLAMA_SERVER=/app/llama.cpp/llama-server
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/app/llama.cpp:${LD_LIBRARY_PATH}

# Install essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip python3.11 python3.11-venv python3-pip libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && ln -sf /usr/bin/python3.11 /usr/bin/python

# Download the official precompiled CUDA 12 binary of llama.cpp (Takes 3 seconds!)
RUN mkdir -p /app/models /app/llama.cpp \
    && wget -q -O /tmp/llama.zip "https://github.com/ggerganov/llama.cpp/releases/download/b3821/llama-b3821-bin-ubuntu-x64.zip" \
    && unzip -j /tmp/llama.zip -d /app/llama.cpp/ \
    && chmod +x /app/llama.cpp/llama-server \
    && rm /tmp/llama.zip

# Install Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir runpod huggingface-hub

# Pre-download GGUF models so cold-start is instant
RUN python3 -c "\
from huggingface_hub import hf_hub_download;\
hf_hub_download('mradermacher/PaddleOCR-VL-1.6-GGUF', 'PaddleOCR-VL-1.6.Q4_K_M.gguf', local_dir='/app/models');\
hf_hub_download('PaddlePaddle/PaddleOCR-VL-1.6-GGUF', 'PaddleOCR-VL-1.6-GGUF-mmproj.gguf', local_dir='/app/models');\
"

WORKDIR /app
COPY . /app

CMD ["python3", "-u", "handler.py"]
