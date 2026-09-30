FROM python:3.11-slim

# Use non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install essential dependencies, clean up immediately to save space
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    unzip \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Paddle, PyMuPDF, RunPod)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir runpod huggingface-hub

# Prepare directories
ENV OCR_CACHE_DIR=/app/models
ENV LLAMA_SERVER=/app/llama.cpp/llama-server
RUN mkdir -p /app/models /app/llama.cpp

# Download llama-server prebuilt CUDA binary (No heavy build-essential or gcc needed!)
RUN wget -q -O /tmp/llama.zip "https://github.com/ggerganov/llama.cpp/releases/download/b3821/llama-b3821-bin-ubuntu-x64.zip" && \
    unzip -j /tmp/llama.zip -d /app/llama.cpp/ && \
    chmod +x /app/llama.cpp/llama-server && \
    rm /tmp/llama.zip

# Pre-download the GGUF models so there is NO COLD START penalty
# We download only the specific Q4_K_M files directly to save disk space
RUN python3 -c "\
from huggingface_hub import hf_hub_download;\
hf_hub_download('mradermacher/PaddleOCR-VL-1.6-GGUF', 'PaddleOCR-VL-1.6.Q4_K_M.gguf', local_dir='/app/models');\
hf_hub_download('PaddlePaddle/PaddleOCR-VL-1.6-GGUF', 'PaddleOCR-VL-1.6-GGUF-mmproj.gguf', local_dir='/app/models');\
"

# Copy the pipeline code
WORKDIR /app
COPY . /app

# Run the handler
CMD ["python3", "-u", "handler.py"]
