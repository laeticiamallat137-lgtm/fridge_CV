# Fine-Tuned vs Final Transfer-Learning Model


## Evaluation Setup

| Setting | Value |
|---|---:|
| Dataset config | `/content/datasets_merges_fridge-1/data.yaml` |
| Images | 1247 |
| Instances | 2477 |
| Image size | 640 |
| Confidence threshold | 0.25 |
| Framework | Ultralytics YOLO |


## Overall Results

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Fine-tuned model | 0.318 | 0.214 | 0.226 | 0.148 |
| Final transfer-learning model | 0.524 | 0.347 | 0.372 | 0.261 |



## Transfer Learning vs Fine-Tuned

| Metric | Difference |
|---|---:|
| Precision | +0.206 |
| Recall | +0.133 |
| mAP50 | +0.146 |
| mAP50-95 | +0.113 |

the final transfer-learning model improves
across all major detection metrics. The largest relative gain is in mAP50-95,
which suggests stronger localization quality across stricter IoU thresholds.

## Final Transfer-Learning Per-Class Results

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 1247 | 2477 | 0.524 | 0.347 | 0.372 | 0.261 |
| apple | 29 | 53 | 0.255 | 0.358 | 0.316 | 0.232 |
| banana | 59 | 155 | 0.624 | 0.268 | 0.272 | 0.189 |
| bell_pepper | 17 | 39 | 0.153 | 0.0256 | 0.0524 | 0.0274 |
| broccoli | 31 | 80 | 0.566 | 0.489 | 0.550 | 0.391 |
| cabbage | 40 | 58 | 0.578 | 0.377 | 0.443 | 0.356 |
| carrot | 51 | 122 | 0.408 | 0.243 | 0.228 | 0.153 |
| cheese | 34 | 93 | 0.437 | 0.269 | 0.256 | 0.204 |
| egg | 61 | 114 | 0.531 | 0.351 | 0.324 | 0.236 |
| juice | 57 | 92 | 0.735 | 0.489 | 0.580 | 0.400 |
| milk | 25 | 33 | 0.570 | 0.424 | 0.484 | 0.342 |
| orange | 73 | 426 | 0.531 | 0.406 | 0.397 | 0.231 |
| pear | 34 | 131 | 0.713 | 0.399 | 0.478 | 0.371 |
| tomato | 76 | 335 | 0.626 | 0.245 | 0.289 | 0.205 |
| water_bottle | 197 | 746 | 0.602 | 0.520 | 0.534 | 0.316 |

