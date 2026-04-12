# Smart Fridge Inventory Detection

**Team:** Mira El Dalati (230878) · Dia Hajjar (233373) · Leticia Mallat (232951)

## What this project does
A computer vision system that analyzes refrigerator images to automatically detect 
and classify food items, estimate their quantities, and identify missing categories.

## Project Structure
smart-fridge/
├── best.pt                  ← download separately (see below)
├── classes.yaml
├── fridge.yaml
├── src/
│   ├── detect.py            ← runs YOLO model on image
│   ├── inventory.py         ← counts items, flags missing classes
│   └── report.py            ← full pipeline + saves annotated image
├── data/
│   ├── train/
│   ├── val/
│   └── test/
└── results/                 ← annotated output images saved here

## Setup
1. Clone the repo
2. Download best.pt from Google Drive: [ADD LINK HERE]
   Place it in the smart-fridge/ folder
3. Install dependencies:
pip install ultralytics opencv-python

## How to run
python src/report.py --image path/to/fridge.jpg

Optional flags:
--conf 0.25       confidence threshold (default 0.25, try 0.1 for more detections)
--output results  output folder for annotated image
--no-save         skip saving annotated image

## Target Classes (15 total)
Fruits: apple, banana, orange, pear
Vegetables: tomato, carrot, broccoli, bell_pepper, cabbage
Dairy & Eggs: egg, milk, yogurt, cheese
Beverages: water_bottle, juice

## Pipeline
detect.py → inventory.py → report.py

## Phases completed
- Phase 1: Class definition (classes.yaml)
- Phase 2: Data collection (Open Images + Roboflow, 7972 images)
- Phase 3: Preprocessing & augmentation
- Phase 4: YOLOv8s training (20 epochs, T4 GPU, Google Colab)
- Phase 5: Counting & inventory logic

## Next steps
- Phase 6: Evaluation (mAP, precision, recall, real fridge test set)
- Phase 7: Report generation & visualization
