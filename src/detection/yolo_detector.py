"""
yolo_detector.py
-----------------
Wraps Ultralytics YOLOv8 for dog detection in images/frames.
Uses pretrained COCO weights (no custom training needed for detection).

COCO class index 16 = "dog"
"""

import torch
from ultralytics import YOLO
import cv2

class DogDetector:
    def __init__(self, model_path="yolov8n.pt", device=None, conf_threshold=0.4):
        """
        Args:
            model_path: path or name of YOLO weights. 'yolov8n.pt' auto-downloads
                        pretrained COCO weights on first run.
            device: 'cuda' or 'cpu'. Auto-detects if None.
            conf_threshold: minimum confidence to keep a detection.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"[DogDetector] Loaded {model_path} on device: {self.device}")

    def detect(self, image_path_or_frame):
        """
        Runs detection and returns only 'dog' class boxes.

        Returns:
            List of dicts: [{"bbox": [x1, y1, x2, y2], "confidence": float}, ...]
        """
        results = self.model(
            image_path_or_frame,
            classes=[16],  # COCO class 16 = dog
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append({"bbox": xyxy, "confidence": conf})

        return detections

    def detect_and_save(self, image_path_or_frame, save_path="output.jpg"):
        """Runs detection and saves an annotated image for visual confirmation."""
        results = self.model(
            image_path_or_frame,
            classes=[16],
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        results[0].save(filename=save_path)
        print(f"[DogDetector] Annotated result saved to: {save_path}")
        return results



if __name__ == "__main__":
    detector = DogDetector()

    detector.detect_and_save(
        "C://Users//LOQ//Downloads//images.jpg",
        save_path="output.jpg"
    )

    img = cv2.imread("output.jpg")
    cv2.imshow("Dog Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # test_image = "C://Users//LOQ//Downloads//images.jpg"  # <-- put a dog photo here to test

    # detections = detector.detect(test_image)
    # print(f"Found {len(detections)} dog(s):")
    # for d in detections:
    #     print(f"  bbox={d['bbox']}, confidence={d['confidence']:.2f}")

    # detector.detect_and_save(test_image, save_path="output.jpg")