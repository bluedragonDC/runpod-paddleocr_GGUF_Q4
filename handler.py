import base64
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import runpod

logging.basicConfig(level=logging.INFO)

pipeline = None
try:
    from ocr_pipeline import get_pipeline
    pipeline = get_pipeline()
    logging.info("OcrPipeline initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize OcrPipeline: {e}\n{traceback.format_exc()}")

def process_job(job: dict[str, Any]) -> dict[str, Any]:
    if pipeline is None:
        return {
            "error": "OcrPipeline failed to initialize on worker startup. Check server logs for details."
        }

    job_input = job.get("input", {})

    pdf_base64 = job_input.get("pdf_base64")
    image_base64 = job_input.get("image")

    if not pdf_base64 and not image_base64:
        return {"error": "Missing 'pdf_base64' or 'image' in input."}

    pages = job_input.get("pages", None)

    try:
        if pdf_base64:
            pdf_bytes = base64.b64decode(pdf_base64, validate=True)
            result = pipeline.convert(pdf_bytes, pages)
            return result

        if image_base64:
            # If the user passes 'data:image/jpeg;base64,...', strip the prefix
            if isinstance(image_base64, str) and image_base64.startswith("data:image/"):
                image_base64 = image_base64.split(",", 1)[1]

            img_bytes = base64.b64decode(image_base64, validate=True)

            # Convert JPEG bytes to PDF stream via pymupdf
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
    if pipeline is not None:
        runpod.serverless.start({"handler": process_job})
    else:
        logging.error("Container failing to start because pipeline is null.")
