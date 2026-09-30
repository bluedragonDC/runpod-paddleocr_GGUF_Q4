"""Small LAN-only-in-practice HTTP test API using the existing local Q4 setup."""

from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OCR_CACHE_DIR", str(ROOT / "models"))
os.environ.setdefault("OCR_MODEL_PATH", str(ROOT / "models" / "PaddleOCR-VL-1.6.Q4_K_M.gguf"))
os.environ.setdefault("OCR_PROJECTOR_PATH", str(ROOT / "models" / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"))
os.environ.setdefault("LLAMA_SERVER", str(ROOT / "tools" / "llama.cpp" / "llama-server"))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / ".cache" / "paddlex"))

from ocr_pipeline import get_pipeline  # noqa: E402


HOST = os.environ.get("OCR_HOST", "0.0.0.0")
PORT = int(os.environ.get("OCR_PORT", "8000"))
MAX_PDF_BYTES = int(os.environ.get("MAX_PDF_BYTES", str(8 * 1024 * 1024)))
MAX_PAGES = int(os.environ.get("HARD_MAX_PAGES", "30"))
MAX_PENDING = int(os.environ.get("LOCAL_MAX_QUEUED_JOBS", "3"))
REQUESTS_PER_MINUTE = int(os.environ.get("LOCAL_REQUESTS_PER_MINUTE", "10"))
API_KEY = os.environ.get("OCR_LOCAL_API_KEY") or ""
REQUIRE_API_KEY = bool(API_KEY)


class QueueFull(Exception):
    pass


class LocalFifo:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.waiters: deque[object] = deque()
        self.active = False

    def acquire(self, timeout: int) -> tuple[object, int]:
        ticket = object()
        deadline = time.monotonic() + timeout
        with self.condition:
            if len(self.waiters) >= MAX_PENDING:
                raise QueueFull("The local OCR queue is full")
            position = len(self.waiters)
            self.waiters.append(ticket)
            while self.active or self.waiters[0] is not ticket:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.waiters.remove(ticket)
                    self.condition.notify_all()
                    raise QueueFull("Timed out waiting for the local GPU slot")
                self.condition.wait(remaining)
            self.waiters.popleft()
            self.active = True
            return ticket, position

    def release(self) -> None:
        with self.condition:
            self.active = False
            self.condition.notify_all()


class LocalRateLimit:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: dict[str, deque[float]] = {}

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self.lock:
            times = self.requests.setdefault(key, deque())
            while times and now - times[0] >= 60:
                times.popleft()
            if len(times) >= REQUESTS_PER_MINUTE:
                raise ValueError("Rate limit reached; wait before sending another request")
            times.append(now)


FIFO = LocalFifo()
RATE_LIMIT = LocalRateLimit()


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "Pdf2MdLocal/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        # Keep credentials and PDF payloads out of request logs.
        print(f"{self.client_address[0]} - {fmt % args}", flush=True)

    def _json(self, status: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok", "service": "paddleocr-vl-pdf2md-local"})
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/v1/ocr":
            self._json(404, {"error": "not_found"})
            return
        supplied = self.headers.get("Authorization", "")
        token = supplied[7:] if supplied.startswith("Bearer ") else ""
        if REQUIRE_API_KEY:
            if not token or not hmac.compare_digest(token, API_KEY):
                self._json(401, {"error": "unauthorized", "message": "Send Authorization: Bearer <API key>"})
                return

        try:
            RATE_LIMIT.check(self.client_address[0])
        except ValueError as exc:
            self._json(429, {"error": "rate_limit_exceeded", "message": str(exc)})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "invalid_content_length"})
            return
        if length < 1 or length > (MAX_PDF_BYTES * 4 // 3 + 64 * 1024):
            self._json(413, {"error": "payload_too_large", "max_pdf_bytes": MAX_PDF_BYTES})
            return
        try:
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("JSON body must be an object")
            encoded = request.get("pdf_base64")
            if not isinstance(encoded, str) or not encoded:
                raise ValueError("pdf_base64 is required")
            if len(encoded) > MAX_PDF_BYTES * 4 // 3 + 16:
                self._json(413, {"error": "payload_too_large", "max_pdf_bytes": MAX_PDF_BYTES})
                return
            try:
                pdf_bytes = base64.b64decode(encoded, validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise ValueError("pdf_base64 is not valid base64") from exc
            if len(pdf_bytes) > MAX_PDF_BYTES:
                self._json(413, {"error": "payload_too_large", "max_pdf_bytes": MAX_PDF_BYTES})
                return
            pages = request.get("pages")
            if pages is not None and (not isinstance(pages, list) or not pages or len(pages) > MAX_PAGES):
                raise ValueError(f"pages must contain 1 to {MAX_PAGES} page numbers")
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._json(400, {"error": "invalid_request", "message": str(exc)})
            return

        try:
            _, queue_position = FIFO.acquire(timeout=int(os.environ.get("LOCAL_QUEUE_WAIT_SECONDS", "240")))
        except QueueFull as exc:
            self._json(429, {"error": "queue_full", "message": str(exc)})
            return
        started = time.perf_counter()
        try:
            result = get_pipeline().convert(pdf_bytes, pages)
            result["request_id"] = secrets.token_hex(12)
            result["queue_position"] = queue_position
            self._json(200, result)
        except ValueError as exc:
            self._json(400, {"error": "invalid_pdf", "message": str(exc)})
        except Exception:
            self._json(500, {"error": "ocr_failed", "message": "OCR failed; inspect the local server log"})
            raise
        finally:
            FIFO.release()
            print(f"OCR request finished in {time.perf_counter() - started:.2f}s", flush=True)


def lan_ip() -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.168.1.1", 80))
        return probe.getsockname()[0]
    except OSError:
        return "<PC-LAN-IP>"
    finally:
        probe.close()


if __name__ == "__main__":
    httpd = ThreadingHTTPServer((HOST, PORT), ApiHandler)
    httpd.daemon_threads = True
    print(f"PDF → Markdown API listening on http://{lan_ip()}:{PORT}", flush=True)
    print(f"API key: {API_KEY}", flush=True)
    print("POST /v1/ocr | GET /health | Ctrl+C to stop", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stopping local API...", flush=True)
    finally:
        httpd.server_close()
