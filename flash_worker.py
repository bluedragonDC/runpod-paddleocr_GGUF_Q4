"""Queued GPU worker for Runpod Flash."""

from __future__ import annotations

import os

from runpod_flash import Endpoint, GpuGroup


shared_env = {
    "REDIS_URL": os.environ.get("REDIS_URL", ""),
    "OCR_CACHE_DIR": os.environ.get("OCR_CACHE_DIR", "/runpod-volume/ocr-cache"),
    "HF_HOME": os.environ.get("HF_HOME", "/runpod-volume/huggingface"),
    "PADDLE_PDX_CACHE_HOME": os.environ.get("PADDLE_PDX_CACHE_HOME", "/runpod-volume/paddlex"),
    "LLAMA_SERVER": os.environ.get("LLAMA_SERVER", "/runpod-volume/llama.cpp/llama-server"),
    "MAX_QUEUE_WAIT_SECONDS": os.environ.get("MAX_QUEUE_WAIT_SECONDS", "240"),
    "TENANT_JOB_LEASE_SECONDS": os.environ.get("TENANT_JOB_LEASE_SECONDS", "1800"),
    "MAX_PDF_BYTES": os.environ.get("MAX_PDF_BYTES", str(8 * 1024 * 1024)),
    "HARD_MAX_PAGES": os.environ.get("HARD_MAX_PAGES", "30"),
}

@Endpoint(
    name="paddleocr-vl-pdf2md-gpu",
    gpu=GpuGroup.ANY,
    workers=(0, 3),
    idle_timeout=60,
    dependencies=[
        "redis==5.2.1",
        "PyMuPDF==1.26.7",
        "huggingface-hub>=0.30",
        "paddleocr==3.7.0",
        "paddlex[ocr]==3.7.2",
        "paddlepaddle==3.2.0",
    ],
    system_dependencies=["libgomp1", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1"],
    env=shared_env,
)
async def process_ocr(payload: dict) -> dict:
    import base64
    import binascii
    import logging
    import threading
    import uuid

    import redis

    from ocr_pipeline import get_pipeline
    from tenant_queue import RedisTenantQueue, TenantQueueFull

    if not isinstance(payload, dict):
        raise ValueError("worker input must be an object")
    tenant_id = str(payload.get("tenant_id", "")).strip()
    encoded = payload.get("pdf_base64")
    if not tenant_id or not isinstance(encoded, str) or not encoded:
        raise ValueError("tenant_id and pdf_base64 are required")
    max_active = payload.get("max_active_jobs", 1)
    max_queued = payload.get("max_queued_jobs", 3)
    max_user_pages = payload.get("max_pages_per_job", 10)
    pages = payload.get("pages")
    if pages is not None and (not isinstance(pages, list) or len(pages) > max_user_pages):
        raise ValueError(f"This API key allows at most {max_user_pages} pages per job")

    try:
        pdf_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("pdf_base64 is not valid base64") from exc
    if len(pdf_bytes) > int(os.environ.get("MAX_PDF_BYTES", str(8 * 1024 * 1024))):
        raise ValueError("PDF payload is too large")

    redis_url = os.environ.get("REDIS_URL", "")
    if not redis_url:
        raise RuntimeError("REDIS_URL is required for the per-user FIFO queue")
    client = redis.Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=5, socket_timeout=5)
    queue = RedisTenantQueue(client, tenant_id)
    request_id = str(uuid.uuid4())
    lease_ms = int(os.environ.get("TENANT_JOB_LEASE_SECONDS", "1800")) * 1000
    try:
        position = queue.enqueue_and_acquire(
            request_id,
            max_active=max_active,
            max_queued=max_queued,
            wait_timeout_s=int(os.environ.get("MAX_QUEUE_WAIT_SECONDS", "240")),
            lease_ms=lease_ms,
        )
    except TenantQueueFull as exc:
        raise RuntimeError(f"user_queue_full: {exc}") from exc
    except redis.RedisError as exc:
        raise RuntimeError("tenant_queue_unavailable: request rejected safely") from exc

    stop = threading.Event()
    logger = logging.getLogger("ocr-api")

    def renew_lease() -> None:
        while not stop.wait(max(5, lease_ms // 3000)):
            try:
                if not queue.renew(request_id, lease_ms):
                    logger.error("Tenant queue lease lost for request %s", request_id)
                    return
            except redis.RedisError:
                logger.exception("Could not renew tenant queue lease")

    heartbeat = threading.Thread(target=renew_lease, daemon=True)
    heartbeat.start()
    try:
        result = get_pipeline().convert(pdf_bytes, pages)
        result["request_id"] = request_id
        result["user_queue_position"] = position
        return result
    finally:
        stop.set()
        heartbeat.join(timeout=2)
        try:
            queue.remove(request_id)
        except redis.RedisError:
            logger.exception("Could not release tenant queue slot for request %s", request_id)
