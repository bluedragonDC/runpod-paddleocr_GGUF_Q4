# PaddleOCR-VL PDF → Markdown API

This project deploys with Runpod Flash, using managed Python runtimes. There is no project Dockerfile and no custom image to build. Flash creates two Serverless endpoints: a small CPU HTTP API and a queued GPU OCR worker.

## What it does

- `POST /v1/ocr`: accepts a base64 PDF and optional 1-based page list; returns Markdown, page and word counts, elapsed seconds, words/second and a request ID.
- `GET /health`: API health response.
- The CPU API checks each customer key and applies a Redis-backed per-minute limit before submitting to the GPU job queue.
- Redis also provides a fair FIFO per-customer queue, active-job cap and pending-job cap across GPU workers.
- PDF cap: 8 MiB. PDF page cap: 30. Customer plans can set lower page caps.
- PaddleOCR-VL keeps headings, footnotes, headers and page numbers in Markdown (`markdown_ignore_labels=[]`).

## Storage and model runtime

The GPU worker expects a Runpod network volume mounted at `/runpod-volume`. It downloads the Q4 GGUF and projector there on first use, into `/runpod-volume/ocr-cache/`, and caches Hugging Face/PaddleX assets on the same volume. Place a CUDA-enabled `llama-server` and its CUDA shared libraries in `/runpod-volume/llama.cpp/`; the default executable path is `/runpod-volume/llama.cpp/llama-server`.

The existing local CUDA build is under `tools/llama.cpp/`. Copy that directory to the GPU worker's network volume if it matches the Runpod GPU runtime. If it does not, prepare a CUDA-compatible llama.cpp build on a Runpod GPU Pod and keep it on the same network volume. Model weights and the runner stay on that network volume, outside the Flash deployment artifact.

## Deploy

1. Create a Redis database with TLS and a 25 GB Runpod network volume in the GPU worker's datacenter. Attach the volume to the GPU endpoint at `/runpod-volume` after Flash creates the endpoints.
2. In the deployment shell, set `RUNPOD_API_KEY`, `REDIS_URL`, and `OCR_API_KEYS_JSON`. Use the template in `.env.example`; replace its key with a generated secret of at least 24 characters. Do not commit real credentials.
3. From this `api` folder, install only the lightweight Flash CLI into the existing project environment, then deploy:

```bash
source ../.venv/bin/activate
python -m pip install -r requirements.txt
flash deploy
```

Flash packages the Python code and endpoint dependencies and deploys onto its managed runtimes. It does not create a custom project image. Flash's deployment artifact has a 1.5 GB limit; if Paddle's GPU dependency set exceeds that, the build will report it before endpoint deployment.

4. Set `OCR_API_KEYS_JSON` and `REDIS_URL` as endpoint environment secrets in Runpod too, so they are present in both the API and GPU worker.
5. Attach and populate the GPU worker's network volume. The first OCR job downloads the two GGUF files if they are not already cached.
6. Set the GPU endpoint execution timeout high enough for the largest PDF you accept. Both endpoints scale from zero to three workers and use a 60-second idle timeout.

## Customer plan example

```json
{
  "replace-with-a-long-random-customer-key-at-least-24-chars": {
    "user_id": "customer-001",
    "requests_per_minute": 10,
    "max_active_jobs": 1,
    "max_queued_jobs": 3,
    "max_pages_per_job": 10
  }
}
```

The HTTP request body is:

```json
{
  "api_key": "replace-with-a-long-random-customer-key-at-least-24-chars",
  "pdf_base64": "JVBERi0xLjQK...",
  "pages": [8, 9]
}
```

## Call the API

Runpod endpoint authorization is required in addition to the customer `api_key`. Keep `RUNPOD_API_KEY` on your own backend; do not put it in a browser or mobile app. Your backend can authenticate its customer, submit this JSON to the Flash API URL, and return the Markdown.

```bash
curl -X POST "$RUNPOD_API_URL/v1/ocr" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @request.json
```

PDF URLs are not fetched by the worker; send base64 or upload via a private backend. This avoids arbitrary server-side URL fetching and SSRF.
