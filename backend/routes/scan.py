from pathlib import Path
import io
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from fastapi import APIRouter, File, UploadFile, HTTPException


router = APIRouter(prefix="/scan", tags=["Detection"])


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

CHECKPOINT_PATH = (
    BASE_DIR
    / "checkpoints"
    / "resnet18_calm_aggressive_best.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# Image preprocessing
# Same preprocessing used during model testing
# ---------------------------------------------------------

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = build_model()

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


# ---------------------------------------------------------
# Image prediction
# ---------------------------------------------------------

def predict_pil_image(image):
    tensor = eval_transform(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(tensor)
        prob_aggressive = torch.sigmoid(logit).item()

    if prob_aggressive >= 0.5:
        prediction = "aggressive"
        confidence = prob_aggressive
    else:
        prediction = "calm"
        confidence = 1 - prob_aggressive

    return prediction, confidence, prob_aggressive


# ---------------------------------------------------------
# IMAGE DETECTION
# ---------------------------------------------------------

@router.post("")
async def scan_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        image = Image.open(
            io.BytesIO(image_data)
        ).convert("RGB")

        prediction, confidence, prob_aggressive = (
            predict_pil_image(image)
        )

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "prob_aggressive": round(prob_aggressive, 4),
            "filename": file.filename,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ---------------------------------------------------------
# VIDEO DETECTION
# ---------------------------------------------------------

@router.post("/video")
async def scan_video(file: UploadFile = File(...)):
    temp_path = None

    try:
        # Read uploaded video
        video_data = await file.read()

        if not video_data:
            raise HTTPException(
                status_code=400,
                detail="Empty video file.",
            )

        # Save video temporarily
        suffix = Path(file.filename or "video.mp4").suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(video_data)
            temp_path = temp_file.name

        # Open video
        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open the uploaded video.",
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30.0

        duration = total_frames / fps if total_frames > 0 else 0

        if total_frames <= 0:
            cap.release()
            raise HTTPException(
                status_code=400,
                detail="Video contains no readable frames.",
            )

        # -------------------------------------------------
        # Sample up to 30 frames from the video
        # -------------------------------------------------

        max_samples = 30

        sample_count = min(
            max_samples,
            total_frames,
        )

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            sample_count,
            dtype=int,
        )

        predictions = []
        confidences = []
        aggressive_probabilities = []

        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index),
            )

            success, frame = cap.read()

            if not success:
                continue

            # OpenCV uses BGR.
            # Convert to RGB for PIL.
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(frame_rgb)

            prediction, confidence, prob_aggressive = (
                predict_pil_image(image)
            )

            predictions.append(prediction)
            confidences.append(confidence)
            aggressive_probabilities.append(
                prob_aggressive
            )

        cap.release()

        if not predictions:
            raise HTTPException(
                status_code=400,
                detail="Could not read any frames from the video.",
            )

        # -------------------------------------------------
        # Combine frame predictions
        # -------------------------------------------------

        aggressive_frames = predictions.count(
            "aggressive"
        )

        calm_frames = predictions.count("calm")

        analyzed_frames = len(predictions)

        aggressive_percentage = (
            aggressive_frames
            / analyzed_frames
        ) * 100

        # Majority voting
        if aggressive_frames > calm_frames:
            final_prediction = "aggressive"
        else:
            final_prediction = "calm"

        # Average confidence
        average_confidence = sum(
            confidences
        ) / len(confidences)

        average_prob_aggressive = sum(
            aggressive_probabilities
        ) / len(aggressive_probabilities)

        return {
            "prediction": final_prediction,
            "confidence": round(
                average_confidence,
                4,
            ),
            "prob_aggressive": round(
                average_prob_aggressive,
                4,
            ),
            "aggressive_frames": aggressive_frames,
            "calm_frames": calm_frames,
            "analyzed_frames": analyzed_frames,
            "aggressive_percentage": round(
                aggressive_percentage,
                2,
            ),
            "duration_seconds": round(
                duration,
                2,
            ),
            "filename": file.filename,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:
        # Delete temporary video file
        if temp_path is not None:
            try:
                Path(temp_path).unlink(
                    missing_ok=True
                )
            except Exception:
                pass