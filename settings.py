"""Runtime configuration for the Runpod OCR worker."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class UserPlan:
    user_id: str
    requests_per_minute: int
    max_active_jobs: int
    max_queued_jobs: int
    max_pages_per_job: int


def load_api_keys() -> dict[str, UserPlan]:
    """Return SHA-256 key hashes mapped to tenant plans."""
    raw = os.environ.get("OCR_API_KEYS_JSON", "")
    if not raw:
        raise RuntimeError("OCR_API_KEYS_JSON is required; refusing to start without API keys")
    try:
        configured = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OCR_API_KEYS_JSON must be a JSON object") from exc
    if not isinstance(configured, dict) or not configured:
        raise RuntimeError("OCR_API_KEYS_JSON must contain at least one API key")

    result: dict[str, UserPlan] = {}
    for secret, limits in configured.items():
        if not isinstance(secret, str) or len(secret) < 24 or not isinstance(limits, dict):
            raise RuntimeError("Each API key must be at least 24 characters and map to a plan object")
        user_id = str(limits.get("user_id", "")).strip()
        if not user_id:
            raise RuntimeError("Every API key plan must set user_id")
        result[hashlib.sha256(secret.encode()).hexdigest()] = UserPlan(
            user_id=user_id,
            requests_per_minute=_positive_int(limits, "requests_per_minute", 10),
            max_active_jobs=_positive_int(limits, "max_active_jobs", 1),
            max_queued_jobs=_positive_int(limits, "max_queued_jobs", 3),
            max_pages_per_job=_positive_int(limits, "max_pages_per_job", 10),
        )
    return result


def _positive_int(values: dict, name: str, default: int) -> int:
    value = values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value
