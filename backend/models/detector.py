"""
Object Detector using YOLOv8x — highest accuracy YOLO model.
"""
import torch
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict

# Project root (yolov8x.pt lives next to backend/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ObjectDetector:
    """
    YOLOv8x-based object detector. Provides bounding boxes,
    confidence scores, and class labels for all detected objects.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        from ultralytics import YOLO
        weights = _PROJECT_ROOT / "yolov8x.pt"
        if not weights.exists():
            weights = "yolov8x.pt"
        self.model = YOLO(str(weights))
        if device in ("cuda", "mps"):
            self.model.to(device)
        print(f"[Detector] YOLOv8x loaded on {device}")

    def detect(self, image: Image.Image, conf_threshold: float = 0.15) -> List[Dict]:
        """
        Run object detection on a PIL Image.
        Returns list of detection dicts with label, confidence, bbox.
        """
        img_np = np.array(image.convert("RGB"))
        img_w, img_h = image.size

        results = self.model(
            img_np,
            conf=conf_threshold,
            iou=0.45,
            imgsz=1280,
            max_det=300,
            augment=True,
            device=self.device,
            verbose=False,
        )
        detections = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    label = result.names[cls_id]

                    area_pct = round(
                        ((x2 - x1) * (y2 - y1)) / (img_w * img_h) * 100, 1
                    )

                    detections.append({
                        "label": label,
                        "confidence": round(conf * 100, 1),
                        "bbox": {
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2)
                        },
                        "area_percent": area_pct,
                        "position": self._describe_position(
                            x1, y1, x2, y2, img_w, img_h
                        )
                    })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def _describe_position(
        self,
        x1: float, y1: float, x2: float, y2: float,
        img_w: int, img_h: int
    ) -> str:
        """Describe the position of the bounding box in natural language."""
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h

        v_pos = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "center")
        h_pos = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")

        if v_pos == "center" and h_pos == "center":
            return "center of frame"
        elif h_pos == "center":
            return f"{v_pos} of frame"
        elif v_pos == "center":
            return f"{h_pos} side of frame"
        else:
            return f"{v_pos}-{h_pos} of frame"
