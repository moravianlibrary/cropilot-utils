# How to train a Cropilot base model

Cropilot’s original page detection models were trained on data exported from ScanTailor projects. These projects were created and labeled by real people, making it a valuable ground truth source.

The training pipeline parses ScanTailor metadata files, extracts page coordinates and related crop information, converts the data into a YOLO-compatible dataset, and uses it to train Cropilot’s own computer vision models.

## Input data

The dataset is created from ScanTailor metadata files and the corresponding source images. Your input folders should follow this structure:

```text
scan-id/
├─ rawdata/
│  ├─ 1/
│  │  └─ <*.tif images>
│  ├─ 2/
│  └─ ...
└─ scanTailor/
   ├─ 1.scanTailor
   ├─ 2.scanTailor
   └─ ...
```

Each `rawdata` subfolder should contain the original TIFF images for a scan batch. The matching `.scanTailor` file contains the crop metadata created in ScanTailor.

## Dataset preparation

### 1. Compress input images

Run:

```bash
base_model_trainer/create_dataset/compress_input_images.py
```

This script converts the input TIFF images into compressed JPG files and saves them in the format expected by the following dataset preparation scripts.

### 2. Extract ScanTailor metadata

Run:

```bash
base_model_trainer/create_dataset/extract_scantailor_data.py
```

This script reads the `.scanTailor` files, extracts crop coordinates and related metadata, and saves them as `metadata.json` files in the corresponding folders.

### 3. Create the YOLO dataset

Run:

```bash
base_model_trainer/create_dataset/create_yolo_dataset.py
```

This script consumes the generated `metadata.json` files and arranges the images and labels into the directory structure expected by Ultralytics YOLO.

See the Ultralytics dataset documentation for details:

https://docs.ultralytics.com/datasets/detect/

During this step, images are padded by 10% on the left and right sides. This allows rotation augmentation to be applied during training without moving page edges outside the image frame.

### 4. Assign classes and clean up labels

Run:

```bash
base_model_trainer/create_dataset/assign_classes_and_cleanup.py
```

This script cleans known issues in the training data and assigns detected objects to Cropilot’s training classes:

- `page`
- `back title cover`
- `unified doublepage`

## Output

The output of the dataset preparation pipeline is a YOLO-compatible dataset that can be used to train the Cropilot page detection model.

The same prepared dataset can also be used as input for the rotate and crop fine-tuning networks.

## Training

Training scripts are stored in:

```text
base_model_trainer.training.crop_train
base_model_trainer.training.rotate_train
```

Both models use the same prepared dataset.

Training reports are periodically saved to Comet ML. To enable Comet ML logging, set the following environment variable before training:

```bash
COMET_ML_API_KEY=<your-api-key>
```