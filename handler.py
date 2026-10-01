import base64
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import runpod

logging.basicConfig(level=logging.INFO)

_pipeline = None


def _get_active_pipeline(job: dict[str, Any]):
    global _pipeline
    if _pipeline is None:
        try:
            runpod.serverless.progress_update(
                job, "🧊 Soğuk Başlangıç: GPU Modelleri VRAM'e Yükleniyor..."
            )
        except Exception:
            pass

        from ocr_pipeline import get_pipeline

        _pipeline = get_pipeline()
        logging.info("OcrPipeline initialized successfully.")
    return _pipeline


def process_job(job: dict[str, Any]) -> dict[str, Any]:
    try:
        pipeline = _get_active_pipeline(job)
    except Exception as e:
        err_msg = f"Failed to initialize OcrPipeline: {e}\n{traceback.format_exc()}"
        logging.error(err_msg)
        return {"error": err_msg}

    job_input = job.get("input", {})

    pdf_base64 = job_input.get("pdf_base64")
    image_base64 = job_input.get("image")

    if not pdf_base64 and not image_base64:
        return {"error": "Missing 'pdf_base64' or 'image' in input."}

    pages = job_input.get("pages", None)

    try:
        try:
            runpod.serverless.progress_update(
                job, "⚡ RunPod GPU: PaddleOCR-VL Çıkarımı Yapılıyor..."
            )
        except Exception:
            pass

        if pdf_base64:
            pdf_bytes = base64.b64decode(pdf_base64, validate=True)
            result = pipeline.convert(pdf_bytes, pages)
            return result

        if image_base64:
            if isinstance(image_base64, str) and image_base64.startswith("data:image/"):
                image_base64 = image_base64.split(",", 1)[1]

            img_bytes = base64.b64decode(image_base64, validate=True)

            import pymupdf

            doc = pymupdf.open()
            img_doc = pymupdf.open("pdf", pymupdf.open("jpeg", img_bytes).convert_to_pdf())
            doc.insert_pdf(img_doc)
            pdf_bytes = doc.write()
            doc.close()
            img_doc.close()

            result = pipeline.convert(pdf_bytes, [1])
            return result

    except Exception as e:
        logging.error(f"Job failed: {str(e)}\n{traceback.format_exc()}")
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": process_job})
