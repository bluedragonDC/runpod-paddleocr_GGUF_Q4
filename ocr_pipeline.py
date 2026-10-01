"""PaddleOCR-VL PDF to structured Markdown pipeline."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pymupdf


def _find_llama_server() -> str:
    env_path = os.environ.get("LLAMA_SERVER")
    if env_path and Path(env_path).is_file():
        return env_path

    found = shutil.which("llama-server")
    if found and Path(found).is_file():
        return found

    candidates = [
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
        "/app/llama.cpp/llama-server",
        "/app/llama.cpp/build/bin/llama-server",
        "/runpod-volume/llama.cpp/llama-server",
    ]
    for cand in candidates:
        if Path(cand).is_file():
            return cand
    return env_path or "/usr/local/bin/llama-server"


CACHE_DIR = Path(os.environ.get("OCR_CACHE_DIR", "/runpod-volume/ocr-cache"))
MODEL = Path(os.environ.get("OCR_MODEL_PATH", str(CACHE_DIR / "PaddleOCR-VL-1.6.Q4_K_M.gguf")))
PROJECTOR = Path(os.environ.get("OCR_PROJECTOR_PATH", str(CACHE_DIR / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf")))
LLAMA_SERVER = _find_llama_server()
LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8112")
MAX_PDF_BYTES = int(os.environ.get("MAX_PDF_BYTES", str(8 * 1024 * 1024)))
MAX_PAGES = int(os.environ.get("HARD_MAX_PAGES", "30"))
RENDER_SCALE = float(os.environ.get("OCR_RENDER_SCALE", "2.0"))


class OcrPipeline:
    def __init__(self):
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        PROJECTOR.parent.mkdir(parents=True, exist_ok=True)
        if not MODEL.is_file() or not PROJECTOR.is_file():
            from huggingface_hub import hf_hub_download

            if not MODEL.is_file():
                hf_hub_download(
                    "mradermacher/PaddleOCR-VL-1.6-GGUF",
                    "PaddleOCR-VL-1.6.Q4_K_M.gguf",
                    local_dir=str(MODEL.parent),
                )
            if not PROJECTOR.is_file():
                hf_hub_download(
                    "PaddlePaddle/PaddleOCR-VL-1.6-GGUF",
                    "PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
                    local_dir=str(PROJECTOR.parent),
                )
        server_path = Path(LLAMA_SERVER)
        if not MODEL.is_file() or not PROJECTOR.is_file() or not server_path.is_file():
            raise RuntimeError(
                f"Missing files - model: {MODEL.is_file()} ({MODEL}), "
                f"projector: {PROJECTOR.is_file()} ({PROJECTOR}), "
                f"llama-server: {server_path.is_file()} ({LLAMA_SERVER})"
            )

        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(server_path.parent), os.environ.get("LD_LIBRARY_PATH", "")]
        )
        self.server_log = open("/tmp/llama-server.log", "ab", buffering=0)
        self.server = subprocess.Popen(
            [
                str(server_path),
                "--model",
                str(MODEL),
                "--mmproj",
                str(PROJECTOR),
                "--host",
                "127.0.0.1",
                "--port",
                "8112",
                "--ctx-size",
                "8192",
                "--parallel",
                "1",
                "--n-gpu-layers",
                "99",
            ],
            env=os.environ.copy(),
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
        )
        self._wait_ready()
        from paddleocr import PaddleOCRVL

        self.pipeline = PaddleOCRVL(
            pipeline_version="v1.6",
            vl_rec_backend="llama-cpp-server",
            vl_rec_server_url=f"{LLAMA_URL}/v1",
            markdown_ignore_labels=[],
        )

    def _wait_ready(self, timeout_s: int = 300) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                log_content = ""
                try:
                    if Path("/tmp/llama-server.log").exists():
                        log_content = Path("/tmp/llama-server.log").read_text(encoding="utf-8")
                except Exception:
                    pass
                raise RuntimeError(
                    f"llama-server stopped during startup. Exit code: {self.server.poll()}. Log: {log_content[-500:]}"
                )
            try:
                with urllib.request.urlopen(f"{LLAMA_URL}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                time.sleep(1)
        raise TimeoutError("llama-server startup timed out")

    def convert(self, pdf_bytes: bytes, requested_pages: list[int] | None) -> dict:
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise ValueError(f"PDF exceeds the {MAX_PDF_BYTES // (1024 * 1024)} MB limit")
        if not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError("Input is not a PDF")

        started = time.perf_counter()
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.is_encrypted:
                raise ValueError("Password-protected PDFs are not supported")
            page_count = len(document)
            if page_count < 1:
                raise ValueError("PDF contains no pages")
            pages = requested_pages or list(range(1, page_count + 1))
            if not isinstance(pages, list) or not pages or len(pages) > MAX_PAGES:
                raise ValueError(f"Select between 1 and {MAX_PAGES} pages")
            if any(
                isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= page_count
                for page in pages
            ):
                raise ValueError(f"Every page number must be between 1 and {page_count}")
            if len(set(pages)) != len(pages):
                raise ValueError("pages cannot contain duplicates")

            markdown_parts = []
            total_words = 0
            for page_number in pages:
                page = document[page_number - 1]
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE), alpha=False
                )
                with tempfile.TemporaryDirectory(prefix="ocr-") as temp_dir:
                    image_path = Path(temp_dir) / f"page-{page_number}.png"
                    output_dir = Path(temp_dir) / "markdown"
                    pixmap.save(str(image_path))
                    results = self.pipeline.predict(str(image_path))
                    result_files = []
                    for result in results:
                        result.save_to_markdown(save_path=str(output_dir))
                    result_files = (
                        list(output_dir.glob("*.md")) if output_dir.exists() else []
                    )
                    if not result_files:
                        raise RuntimeError(f"OCR produced no Markdown for page {page_number}")
                    content = (
                        max(result_files, key=lambda path: path.stat().st_mtime)
                        .read_text(encoding="utf-8")
                        .strip()
                    )
                total_words += len(content.split())
                markdown_parts.append(f"<!-- Page {page_number} -->\n\n{content}")

        elapsed = time.perf_counter() - started
        return {
            "markdown": "\n\n---\n\n".join(markdown_parts),
            "pages_processed": pages,
            "total_pages": page_count,
            "word_count": total_words,
            "processing_seconds": round(elapsed, 3),
            "words_per_second": round(total_words / elapsed, 2) if elapsed else 0,
            "model": "PaddleOCR-VL-1.6 Q4_K_M",
        }


_pipeline: OcrPipeline | None = None


def get_pipeline() -> OcrPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = OcrPipeline()
    return _pipeline
