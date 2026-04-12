# Smart Fridge Inventory Detection

**Team:** Mira El Dalati (230878), Dia Hajjar (233373), Leticia Mallat (232951)

## Description

Smart Fridge Inventory Detection is a computer vision system that automatically identifies and tracks food items inside a refrigerator. Given an image of a fridge, the system detects 15 categories of common food items (fruits, vegetables, dairy, and beverages), generates an inventory report, and helps users keep track of what they have on hand.

## Setup

Install the required dependencies:

```bash
pip install ultralytics opencv-python
```

## How to Run

```bash
python src/report.py --image path/to/fridge.jpg
```

## Model Weights

> **Note:** `best.pt` is not included in this repository.
> You must download it from Google Drive and place it in the root folder (`smart-fridge/best.pt`) before running the project.

## Project Structure

```
smart-fridge/
├── best.pt                  ← download separately and place here
├── classes.yaml             ← class names and count
├── fridge.yaml              ← dataset config for YOLO training
├── .gitignore
├── README.md
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── src/
│   ├── detect.py            ← runs YOLO inference on an image
│   ├── inventory.py         ← parses detections into an inventory list
│   └── report.py            ← entry point: generates the final report
└── results/                 ← output images and reports saved here
```
