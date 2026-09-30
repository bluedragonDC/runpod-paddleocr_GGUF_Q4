import base64
import json
import logging
from pathlib import Path
from typing import Any

import runpod

# Initialize the pipeline once when the container starts
try:
    from ocr_pipeline import get_pipeline
    pipeline = get_pipeline()
    logging.info("OcrPipeline initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize OcrPipeline: {e}")
    pipeline = None

def process_job(job: dict[str, Any]) -> dict[str, Any]:
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

            # Since pipeline expects PDF bytes by default, we need to wrap the image in a PDF
            # A cleaner approach is calling the predictor directly for a single image
            # But to keep it exactly identical to the local logic, we create a temporary PDF:
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
        logging.error(f"Job failed: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    if pipeline is not None:
        runpod.serverless.start({"handler": process_job})
    else:
        logging.error("Container failing to start because pipeline is null.")
