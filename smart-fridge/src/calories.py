"""
Calorie estimation helpers for detected fridge items.

Calories are approximate. For object-sized foods, the estimate is adjusted using
the bounding-box area as a rough size signal. Packaged/liquid items use a fixed
baseline estimate.
"""

from __future__ import annotations

import math


CALORIE_BASE = {
    "apple": 95,
    "banana": 105,
    "orange": 62,
    "pear": 101,
    "tomato": 22,
    "carrot": 25,
    "broccoli": 50,
    "bell_pepper": 31,
    "cabbage": 200,
    "egg": 78,
    "milk": 150,
    "yogurt": 120,
    "cheese": 113,
    "water_bottle": 0,
    "juice": 110,
}

REFERENCE_AREA = {
    "apple": 35000,
    "banana": 50000,
    "orange": 30000,
    "pear": 35000,
    "tomato": 22000,
    "carrot": 18000,
    "broccoli": 40000,
    "bell_pepper": 28000,
    "cabbage": 70000,
    "egg": 12000,
    "milk": 45000,
    "yogurt": 25000,
    "cheese": 22000,
    "water_bottle": 40000,
    "juice": 35000,
}

SIZE_ADJUSTED_CLASSES = {
    "apple",
    "banana",
    "orange",
    "pear",
    "tomato",
    "carrot",
    "broccoli",
    "bell_pepper",
    "cabbage",
    "cheese",
    "yogurt",
}


def estimate_object_calories(class_name: str, x1: float, y1: float, x2: float, y2: float) -> float:
    base = CALORIE_BASE.get(class_name, 0)
    if class_name not in SIZE_ADJUSTED_CLASSES:
        return round(base, 1)

    ref_area = REFERENCE_AREA.get(class_name, 30000)
    box_area = max((x2 - x1), 1) * max((y2 - y1), 1)
    size_factor = math.sqrt(box_area / ref_area)
    size_factor = max(0.7, min(size_factor, 1.4))
    return round(base * size_factor, 1)


def estimate_detections_calories(detections: list[dict]) -> dict:
    items = []
    total = 0.0

    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["bbox"]
        class_name = detection["class_name"]
        calories = estimate_object_calories(class_name, x1, y1, x2, y2)
        total += calories
        items.append(
            {
                "index": index,
                "class_name": class_name,
                "confidence": detection["confidence"],
                "calories": calories,
                "bbox": detection["bbox"],
            }
        )

    return {"items": items, "total_calories": round(total, 1)}
