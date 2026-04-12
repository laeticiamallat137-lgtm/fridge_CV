"""
report.py
---------
Phase 5 — Smart Fridge Inventory Detection
Ties the full pipeline together:
  1. Runs detection on a fridge image
  2. Builds the inventory
  3. Saves an annotated image with bounding boxes
  4. Prints a structured inventory report

Usage:
    python report.py --image path/to/fridge.jpg
    python report.py --image path/to/fridge.jpg --output results/
    python report.py --image path/to/fridge.jpg --conf 0.3 --no-save
"""

import argparse
import os
import cv2

from detect    import run_detection
from inventory import build_inventory, get_grouped_counts, ALL_CLASSES

# ── Colors per food group (BGR for OpenCV) ────────────────────────────────────
GROUP_COLORS = {
    "apple":        (50,  200,  50),   # green
    "banana":       (0,   200, 255),   # yellow
    "orange":       (0,   140, 255),   # orange
    "pear":         (80,  220,  80),   # light green
    "tomato":       (50,   50, 220),   # red
    "carrot":       (0,   120, 255),   # deep orange
    "broccoli":     (20,  160,  20),   # dark green
    "bell_pepper":  (30,   30, 200),   # red-ish
    "cabbage":      (100, 200, 100),   # pale green
    "egg":          (180, 180, 255),   # pale pink
    "milk":         (220, 220, 220),   # white/gray
    "yogurt":       (200, 180, 255),   # lavender
    "cheese":       (0,   200, 240),   # gold
    "water_bottle": (255, 180,  50),   # blue
    "juice":        (0,   180, 255),   # amber
}
DEFAULT_COLOR = (200, 200, 200)
# ─────────────────────────────────────────────────────────────────────────────


def draw_boxes(image_path: str, detections: list[dict], output_path: str) -> str:
    """
    Draw bounding boxes and labels on the image and save it.

    Args:
        image_path:  path to original image
        detections:  list of detection dicts from detect.run_detection()
        output_path: directory to save the annotated image

    Returns:
        Path to the saved annotated image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        name  = det["class_name"]
        conf  = det["confidence"]
        color = GROUP_COLORS.get(name, DEFAULT_COLOR)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

        # Draw label background
        label     = f"{name} {conf:.2f}"
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness  = 1
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)

        # Draw label text
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Save annotated image
    os.makedirs(output_path, exist_ok=True)
    base_name    = os.path.splitext(os.path.basename(image_path))[0]
    save_path    = os.path.join(output_path, f"{base_name}_annotated.jpg")
    cv2.imwrite(save_path, img)

    return save_path


def print_report(inventory: dict, image_path: str, save_path: str = None) -> None:
    """
    Print a structured inventory report to the terminal.
    """
    grouped = get_grouped_counts(inventory)
    divider = "=" * 50

    print(f"\n{divider}")
    print("   SMART FRIDGE INVENTORY REPORT")
    print(divider)
    print(f"  Image : {os.path.basename(image_path)}")
    if save_path:
        print(f"  Saved : {save_path}")
    print(f"  Total items detected: {inventory['total']}")
    print(divider)

    # Detected items by group
    print("\n  DETECTED ITEMS\n")
    for group, items in grouped.items():
        detected_in_group = {k: v for k, v in items.items() if v > 0}
        if detected_in_group:
            print(f"  {group}:")
            for cls, count in detected_in_group.items():
                bar = "█" * count
                print(f"    {cls:<15} {count}x  {bar}")
            print()

    # Missing categories
    if inventory["missing"]:
        print(f"  MISSING CATEGORIES ({len(inventory['missing'])} not detected)\n")
        for cls in inventory["missing"]:
            print(f"    - No {cls} detected")
    else:
        print("  All 15 categories detected!")

    print(f"\n{divider}\n")


def run_pipeline(image_path: str,
                 output_dir: str = "results",
                 conf: float = 0.25,
                 save: bool = True) -> dict:
    """
    Run the full pipeline: detect → inventory → report.

    Args:
        image_path: path to fridge image
        output_dir: where to save annotated image
        conf:       confidence threshold
        save:       whether to save the annotated image

    Returns:
        inventory dict from build_inventory()
    """
    # Step 1: detect
    print(f"Running detection on {image_path}...")
    detections = run_detection(image_path, conf=conf)
    print(f"  Found {len(detections)} detection(s).")

    # Step 2: inventory
    inventory = build_inventory(detections)

    # Step 3: annotate and save
    save_path = None
    if save and detections:
        save_path = draw_boxes(image_path, detections, output_dir)
        print(f"  Annotated image saved to: {save_path}")

    # Step 4: print report
    print_report(inventory, image_path, save_path)

    return inventory


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Fridge Inventory Report")
    parser.add_argument("--image",   required=True,          help="Path to fridge image")
    parser.add_argument("--output",  default="results",      help="Output directory for annotated image")
    parser.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-save", action="store_true",    help="Skip saving annotated image")
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        output_dir=args.output,
        conf=args.conf,
        save=not args.no_save,
    )
