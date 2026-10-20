import torch
from PIL import Image
from torchvision import transforms

from src.detection.yolo_detector import DogDetector
from src.classification.model import load_model

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def get_risk_level(prob_aggressive: float) -> tuple:
    if prob_aggressive < 0.35:
        return "Low", round(1.0 - prob_aggressive, 3)
    elif prob_aggressive < 0.65:
        return "Medium", round(1.0 - abs(prob_aggressive - 0.5) * 2, 3)
    else:
        return "High", round(prob_aggressive, 3)


class InferencePipeline:
    def __init__(self, checkpoint_path: str, yolo_weights: str = "yolov8s.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = DogDetector(model_path=yolo_weights, device=str(self.device))
        self.classifier = load_model(checkpoint_path, self.device)

    def run(self, pil_img: Image.Image) -> dict:
        # Step 1: YOLO
        detections = self.detector.detect(pil_img)

        if not detections:
            return {
                "detected": False,
                "risk_level": "Unknown",
                "confidence": 0.0,
                "prob_aggressive": 0.0,
                "bbox": None,
                "message": "No dog detected in frame",
            }

        # Best detection
        best = max(detections, key=lambda d: d["confidence"])
        bbox = best["bbox"]

        # Step 2: Crop
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cropped = pil_img.crop((x1, y1, x2, y2)).convert("RGB")

        # Step 3: ResNet18
        tensor = TRANSFORM(cropped).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.classifier(tensor).squeeze()
            prob_aggressive = torch.sigmoid(logit).item()

        # Step 4: Risk level
        risk_level, confidence = get_risk_level(prob_aggressive)

        return {
            "detected": True,
            "risk_level": risk_level,
            "confidence": confidence,
            "prob_aggressive": round(prob_aggressive, 3),
            "bbox": [round(v, 1) for v in bbox],
        }