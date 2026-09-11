import shutil
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File

from src.inference.predict import load_models, predict

import base64
import io
import numpy as np
from PIL import Image

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier, segmentation_model = load_models(
    classifier_path="models/classification/weights/classifier_weighted_best.pt",
    segmentation_path="models/segmentation/weights/unet_best.pt",
    device=device,
)

UPLOAD_DIR = Path("uploads_temp")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def read_root():
    return {"status": "Server is running", "device": str(device)}


def mask_to_base64_overlay(mask: np.ndarray) -> str:
    height, width = mask.shape

    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[mask > 0] = [255, 70, 70, 140]

    image = Image.fromarray(overlay, mode="RGBA")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def image_to_base64(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = UPLOAD_DIR / temp_filename

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict(
        image_path=str(temp_path),
        classifier=classifier,
        segmentation_model=segmentation_model,
        device=device,
    )

    response = {
        "has_tumor": result["has_tumor"],
        "confidence": round(result["confidence"], 4),
        "mask_overlay": None,
        "original_image": image_to_base64(str(temp_path)),
    }

    if result["mask"] is not None:
        response["mask_overlay"] = mask_to_base64_overlay(result["mask"])

    temp_path.unlink()

    return response