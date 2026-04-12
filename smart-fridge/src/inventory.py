"""
inventory.py
------------
Phase 5 — Smart Fridge Inventory Detection
Takes raw detections from detect.py, counts items per class,
and identifies missing categories.

Usage:
    from inventory import build_inventory
    inventory = build_inventory(detections)
"""

# ── All 15 target classes (must match classes.yaml) ───────────────────────────
ALL_CLASSES = [
    "apple", "banana", "orange", "pear",          # Fruits
    "tomato", "carrot", "broccoli",                # Vegetables
    "bell_pepper", "cabbage",                      # Vegetables (cont.)
    "egg", "milk", "yogurt", "cheese",             # Dairy & Eggs
    "water_bottle", "juice",                       # Beverages
]

# Group labels for the report
CLASS_GROUPS = {
    "Fruits":       ["apple", "banana", "orange", "pear"],
    "Vegetables":   ["tomato", "carrot", "broccoli", "bell_pepper", "cabbage"],
    "Dairy & Eggs": ["egg", "milk", "yogurt", "cheese"],
    "Beverages":    ["water_bottle", "juice"],
}
# ─────────────────────────────────────────────────────────────────────────────


def build_inventory(detections: list[dict]) -> dict:
    """
    Build an inventory summary from a list of detections.

    Args:
        detections: list of detection dicts from detect.run_detection()

    Returns:
        {
            "counts":   {"apple": 2, "tomato": 3, ...},  # all 15 classes
            "detected": ["apple", "tomato", ...],         # classes with count > 0
            "missing":  ["yogurt", "cabbage", ...],       # classes with count = 0
            "total":    int,                              # total items detected
        }
    """
    # Initialize all classes to 0
    counts = {cls: 0 for cls in ALL_CLASSES}

    # Count detections per class
    for det in detections:
        name = det["class_name"]
        if name in counts:
            counts[name] += 1
        else:
            # Handle case where model detects a class not in our list
            counts[name] = counts.get(name, 0) + 1

    detected = [cls for cls in ALL_CLASSES if counts.get(cls, 0) > 0]
    missing  = [cls for cls in ALL_CLASSES if counts.get(cls, 0) == 0]
    total    = sum(counts[cls] for cls in ALL_CLASSES)

    return {
        "counts":   counts,
        "detected": detected,
        "missing":  missing,
        "total":    total,
    }


def get_grouped_counts(inventory: dict) -> dict:
    """
    Returns counts organized by food group for display.

    Args:
        inventory: output of build_inventory()

    Returns:
        {
            "Fruits":       {"apple": 2, "banana": 0, ...},
            "Vegetables":   {...},
            ...
        }
    """
    grouped = {}
    for group, classes in CLASS_GROUPS.items():
        grouped[group] = {cls: inventory["counts"].get(cls, 0) for cls in classes}
    return grouped


# ── CLI entry point (for quick testing) ──────────────────────────────────────
if __name__ == "__main__":
    # Example: test with dummy detections
    dummy_detections = [
        {"class_id": 0,  "class_name": "apple",   "confidence": 0.91, "bbox": [10, 10, 80, 80]},
        {"class_id": 0,  "class_name": "apple",   "confidence": 0.87, "bbox": [90, 10, 160, 80]},
        {"class_id": 4,  "class_name": "tomato",  "confidence": 0.78, "bbox": [200, 50, 260, 110]},
        {"class_id": 9,  "class_name": "egg",     "confidence": 0.95, "bbox": [300, 20, 340, 60]},
        {"class_id": 9,  "class_name": "egg",     "confidence": 0.93, "bbox": [350, 20, 390, 60]},
        {"class_id": 9,  "class_name": "egg",     "confidence": 0.88, "bbox": [400, 20, 440, 60]},
        {"class_id": 10, "class_name": "milk",    "confidence": 0.82, "bbox": [450, 10, 520, 150]},
    ]

    inventory = build_inventory(dummy_detections)
    grouped   = get_grouped_counts(inventory)

    print("\n=== Inventory Test ===")
    print(f"Total items detected: {inventory['total']}\n")

    for group, items in grouped.items():
        print(f"{group}:")
        for cls, count in items.items():
            status = f"{count} detected" if count > 0 else "-- not found"
            print(f"  {cls:<15} {status}")
        print()

    print("Missing categories:")
    for cls in inventory["missing"]:
        print(f"  - No {cls} detected")
