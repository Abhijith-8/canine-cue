"""
app.py
-------
Flask backend for CanineCue online mode.
Endpoints:
  POST /predict        → single image
  POST /predict/video  → video file (frame-by-frame)
  GET  /health         → server status check
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import cv2
import numpy as np
import tempfile

from src.pipeline.inference_pipeline import InferencePipeline

# ── Init ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

CHECKPOINT = "checkpoints/resnet18_calm_aggressive_v3.pt"
YOLO_WEIGHTS = "yolov8s.pt"

print("[Backend] Loading pipeline...")
pipeline = InferencePipeline(
    checkpoint_path=CHECKPOINT,
    yolo_weights=YOLO_WEIGHTS,
)
print("[Backend] Pipeline ready.")


# ── Health check ──────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "v3"})


# ── Single image predict ───────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_bytes = request.files["image"].read()

    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    result = pipeline.run(pil_img)
    return jsonify(result)


# ── Video predict ──────────────────────────────────────────────
@app.route("/predict/video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    video_file = request.files["video"]

    # Save to temp file (cv2 needs a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Sample max 30 evenly spaced frames
        sample_count = min(30, total_frames)
        if total_frames > 0:
            indices = [int(i * total_frames / sample_count)
                       for i in range(sample_count)]
        else:
            indices = []

        frame_results = []
        aggressive_count = 0
        calm_count = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # cv2 BGR → PIL RGB
            pil_frame = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            result = pipeline.run(pil_frame)
            frame_results.append({
                "frame_index": idx,
                **result
            })

            if result["detected"]:
                if result["risk_level"] == "High":
                    aggressive_count += 1
                else:
                    calm_count += 1

        cap.release()
        os.unlink(tmp_path)

        analyzed = len(frame_results)
        detected = [r for r in frame_results if r["detected"]]

        # Overall verdict: majority risk level among detected frames
        if not detected:
            overall_risk = "Unknown"
            overall_confidence = 0.0
        else:
            high = sum(1 for r in detected if r["risk_level"] == "High")
            medium = sum(1 for r in detected if r["risk_level"] == "Medium")
            low = sum(1 for r in detected if r["risk_level"] == "Low")

            if high >= len(detected) * 0.4:
                overall_risk = "High"
            elif medium >= len(detected) * 0.4:
                overall_risk = "Medium"
            else:
                overall_risk = "Low"

            overall_confidence = round(
                sum(r["confidence"] for r in detected) / len(detected), 3
            )

        return jsonify({
            "overall_risk": overall_risk,
            "overall_confidence": overall_confidence,
            "analyzed_frames": analyzed,
            "detected_frames": len(detected),
            "high_risk_frames": sum(1 for r in detected if r["risk_level"] == "High"),
            "medium_risk_frames": sum(1 for r in detected if r["risk_level"] == "Medium"),
            "low_risk_frames": sum(1 for r in detected if r["risk_level"] == "Low"),
            "frame_results": frame_results,
        })

    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)