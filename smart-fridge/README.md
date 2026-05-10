# Smart Fridge Inventory Detection

**Team:** Mira El Dalati (230878) - Dia Hajjar (233373) - Leticia Mallat (232951)

## What this project does
A computer vision system that analyzes refrigerator images to automatically detect
and classify food items, estimate their quantities, and identify missing categories.

## Project Structure

```text
smart-fridge/
|-- fine_tuned.pt                   # original fine-tuned YOLO model weights
|-- classes.yaml                     # class id/name mapping
|-- fridge.yaml                      # YOLO dataset config
|-- README.md
|-- data/
|   |-- train/
|   |   |-- images/
|   |   `-- labels/
|   |-- val/
|   |   |-- images/
|   |   `-- labels/
|   `-- test/
|       |-- images/
|       `-- labels/
|-- results/
|   |-- evaluation.md
|   `-- model_metrics_comparison.md
`-- src/
    |-- detect.py                   # runs YOLO detection on an image
    |-- inventory.py                # counts items and flags missing classes
    |-- calories.py                 # estimates calories from detections
    |-- report.py                   # full pipeline and annotated output
    |-- web_app.py                  # Streamlit web interface
    |-- evaluate_old_model.py       # evaluates fine_tuned.pt on the transfer-learning dataset
    |-- test11s.py
    |-- transfer_learning_(1) (1)_last.ipynb
    |-- transfer_learning_Plus_calories_count.ipynb
    |-- eggs_test.jpg
    `-- results/
        `-- eggs_test_annotated.jpg
```

## Setup
1. Clone the repo.
2. Keep `fine_tuned.pt` locally in the `smart-fridge/` folder. This is the
   original model trained before the transfer-learning experiment.
3. Keep `best.pt` in Google Drive. This is the final transfer-learning model:
   https://drive.google.com/drive/folders/1Fj8OwVqaA65UzXOVkVOdk4tsy9Jeey4s
4. Install dependencies:

```bash
pip install ultralytics opencv-python pyyaml streamlit pandas
```

## How to run

```bash
python src/report.py --image path/to/fridge.jpg
```

Optional flags:

```text
--conf 0.25        confidence threshold (default 0.25, try 0.1 for more detections)
--output results   output folder for annotated image
--no-save          skip saving annotated image
```

## Web UI

Run the Streamlit interface from the `smart-fridge/` folder:

```bash
streamlit run src/web_app.py
```

The web UI lets you upload a fridge image, run detection, view the inventory,
and press **Calculate calories** to estimate total calories for the detected
items.

The web UI uses the transfer-learning model by default. It looks for `best.pt`
first. If `best.pt` is not in the project folder, set the model path manually:

```bash
set SMART_FRIDGE_MODEL=C:\path\to\best.pt
streamlit run src/web_app.py
```

On Colab or Linux/macOS:

```bash
export SMART_FRIDGE_MODEL=/content/drive/MyDrive/CV_models/fridge_phase2/weights/best.pt
streamlit run src/web_app.py
```


## Target Classes (15 total)
Fruits: apple, banana, orange, pear

Vegetables: tomato, carrot, broccoli, bell_pepper, cabbage

Dairy and Eggs: egg, milk, yogurt, cheese

Beverages: water_bottle, juice
