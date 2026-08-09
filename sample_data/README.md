# Sample Data

A sample of the maize tassel dataset (100 training + 30 validation image-label
pairs) for quick smoke tests of the training pipeline. Images are renamed with
random 4-digit identifiers (tassel_XXXX.jpg) and do not follow the original naming.

- `images/` - sample images
- `labels/` - YOLO-format labels (class 0 = tassel, normalized coordinates)

The full dataset is not included in this repository. To reproduce the reported
results, use the complete dataset and configure `data.yaml` accordingly.
