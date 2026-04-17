"""
detect.py
---------
Phase 5 — Smart Fridge Inventory Detection
Runs the trained YOLOv8 model on a fridge image and returns raw detections.

Usage:
    python detect.py --image path/to/fridge.jpg
    python detect.py --image path/to/fridge.jpg --conf 0.3
"""

import argparse
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best11s.pt"
CONF_THRESH = 0.25            # confidence threshold (0.25 is YOLO default)
IOU_THRESH  = 0.35            # NMS IoU threshold (lowered to catch overlapping items)
IMAGE_SIZE  = 640             # must match training image size
# ─────────────────────────────────────────────────────────────────────────────


def run_detection(image_path: str, conf: float = CONF_THRESH) -> list[dict]:
    """
    Run YOLOv8 detection on a single fridge image.

    Args:
        image_path: path to the input image
        conf:       confidence threshold

    Returns:
        List of detection dicts, each with:
            {
                "class_id":    int,
                "class_name":  str,
                "confidence":  float,
                "bbox":        [x1, y1, x2, y2]  (pixel coords)
            }
    """

    print("Model path:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=image_path,
        conf=conf,
        iou=IOU_THRESH,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )

    detections = []
    for result in results:
        for box in result.boxes:
            class_id   = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class_id":   class_id,
                "class_name": class_name,
                "confidence": round(confidence, 3),
                "bbox":       [x1, y1, x2, y2],
            })

    return detections


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection on a fridge image.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--conf",  type=float, default=CONF_THRESH, help="Confidence threshold")
    args = parser.parse_args()

    print(f"\nRunning detection on: {args.image}")
    print(f"Confidence threshold: {args.conf}")
    print("-" * 40)

    detections = run_detection(args.image, conf=args.conf)

    if not detections:
        print("No objects detected.")
    else:
        print(f"Found {len(detections)} detection(s):\n")
        for d in detections:
            print(f"  [{d['class_id']:>2}] {d['class_name']:<15} "
                  f"conf={d['confidence']:.2f}  bbox={d['bbox']}")
