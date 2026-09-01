"""
NIH ChestX-ray14 Optional FastAPI Research Backend
--------------------------------------------------
REST API providing programmatic inference and Grad-CAM explainability endpoints.
"""

import sys
import io
import base64
from pathlib import Path
from typing import Optional
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.inference_service import run_inference
from app.services.explanation_service import generate_gradcam_explanation
from ml.preprocessing.labels import PATHOLOGY_CLASSES

app = FastAPI(
    title="NIH ChestX-ray14 Research API",
    description="Multi-label chest X-ray inference and visual explainability API backend.",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "NIH ChestX-ray14 Research API",
        "disclaimer": "RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS"
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        result = run_inference(pil_img)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain_endpoint(
    file: UploadFile = File(...),
    target_class: Optional[str] = Form(None)
):
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        exp_res = generate_gradcam_explanation(pil_img, target_class=target_class)

        # Convert overlay PIL image to Base64 JPEG string
        buffered = io.BytesIO()
        exp_res["overlay_pil"].save(buffered, format="JPEG")
        overlay_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "target_class": exp_res["target_class"],
            "target_probability": exp_res["target_probability"],
            "overlay_base64_jpeg": overlay_b64,
            "disclaimer": exp_res["disclaimer"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
