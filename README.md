## Skin Defects Classifier (PyTorch)

This project trains a CNN to classify skin defects using images listed in `skin_defects.csv`. The current dataset contains 3 classes: `acne`, `bags`, `redness`. The pipeline is designed to scale to 10 classes as more labeled data is added.

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the folder structure matches:
```
files/
  acne/0|1|.../front.jpg,left_side.jpg,right_side.jpg
  bags/10|.../front.jpg,left_side.jpg,right_side.jpg
  redness/20|.../front.jpg,left_side.jpg,right_side.jpg
skin_defects.csv
```

### Train

```bash
python train.py --csv skin_defects.csv --files_root files --epochs 15 --batch_size 16 --image_size 224
```

Artifacts are written to `artifacts/`:
- `model_best.pth`
- `model_last.pth`
- `class_to_idx.json`

### Inference

```bash
python inference.py --checkpoint artifacts/model_best.pth --image files/acne/0/front.jpg
```

Output:
```json
{
  "predicted": "acne",
  "confidence": 0.9731,
  "probs": {"acne": 0.9731, "bags": 0.0201, "redness": 0.0068}
}
```

### Notes

- The CSV includes 3 views per person (`front`, `left_side`, `right_side`). The trainer treats each view as an individual sample with the same class label.
- To support 10 diseases, add folders and rows to the CSV with new `type` values; the scripts will adapt automatically.

