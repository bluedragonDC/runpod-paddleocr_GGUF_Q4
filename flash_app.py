"""Runpod Flash API gateway. GPU inference runs on the separate queued worker."""

from __future__ import annotations

import os

from runpod_flash import Endpoint


shared_env = {
    "REDIS_URL": os.environ.get("REDIS_URL", ""),
    "OCR_API_KEYS_JSON": os.environ.get("OCR_API_KEYS_JSON", "{}"),
    "OCR_CACHE_DIR": os.environ.get("OCR_CACHE_DIR", "/runpod-volume/ocr-cache"),
    "HF_HOME": os.environ.get("HF_HOME", "/runpod-volume/huggingface"),
    "PADDLE_PDX_CACHE_HOME": os.environ.get("PADDLE_PDX_CACHE_HOME", "/runpod-volume/paddlex"),
    "LLAMA_SERVER": os.environ.get("LLAMA_SERVER", "/runpod-volume/llama.cpp/llama-server"),
    "MAX_QUEUE_WAIT_SECONDS": os.environ.get("MAX_QUEUE_WAIT_SECONDS", "240"),
    "TENANT_JOB_LEASE_SECONDS": os.environ.get("TENANT_JOB_LEASE_SECONDS", "1800"),
    "MAX_PDF_BYTES": os.environ.get("MAX_PDF_BYTES", str(8 * 1024 * 1024)),
    "HARD_MAX_PAGES": os.environ.get("HARD_MAX_PAGES", "30"),
}

api = Endpoint(
    name="paddleocr-vl-pdf2md-api",
    cpu="cpu5c-4-8",
    workers=(1, 3),
    idle_timeout=60,
    dependencies=["redis==5.2.1"],
    env=shared_env,
)


@api.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "paddleocr-vl-pdf2md"}


@api.post("/v1/ocr")
async def convert_pdf(api_key: str, pdf_base64: str, pages: list[int] | None = None) -> dict:
    import hashlib
    import hmac

    import redis

    from settings import load_api_keys
    from tenant_queue import RateLimited, RedisTenantQueue

    if not isinstance(api_key, str) or not api_key:
        raise ValueError("api_key is required")
    if not isinstance(pdf_base64, str) or not pdf_base64:
        raise ValueError("pdf_base64 is required")

    try:
        plans = load_api_keys()
    except RuntimeError as exc:
        raise RuntimeError("OCR_API_KEYS_JSON is not configured on this endpoint") from exc
    candidate = hashlib.sha256(api_key.encode()).hexdigest()
    plan = next((p for key_hash, p in plans.items() if hmac.compare_digest(candidate, key_hash)), None)
    if plan is None:
        raise PermissionError("Invalid API key")
    if pages is not None and (not isinstance(pages, list) or not pages or len(pages) > plan.max_pages_per_job):
        raise ValueError(f"pages must contain 1 to {plan.max_pages_per_job} page numbers")
    if len(pdf_base64) > int(os.environ.get("MAX_PDF_BYTES", str(8 * 1024 * 1024))) * 4 // 3 + 16:
        raise ValueError("PDF payload is too large")

    redis_url = os.environ.get("REDIS_URL", "")
    if not redis_url:
        raise RuntimeError("REDIS_URL is required for shared rate limits and per-user queue")
    client = redis.Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=5, socket_timeout=5)
    tenant = RedisTenantQueue(client, plan.user_id)
    try:
        tenant.check_rate(plan.requests_per_minute)
    except RateLimited as exc:
        raise RuntimeError(f"rate_limit_exceeded: retry after {exc.retry_after_ms} ms") from exc
    except redis.RedisError as exc:
        raise RuntimeError("tenant_queue_unavailable: request rejected safely") from exc

    from flash_worker import process_ocr

    return await process_ocr({
        "tenant_id": plan.user_id,
        "max_active_jobs": plan.max_active_jobs,
        "max_queued_jobs": plan.max_queued_jobs,
        "max_pages_per_job": plan.max_pages_per_job,
        "pdf_base64": pdf_base64,
        "pages": pages,
    })
