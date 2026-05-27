# Cropilot API tools

Cropilot API tools is a set of scripts to communicate with the Cropilot web application.

## `uploader.py`: Upload and crop images in bulk

The `uploader.py` script allows you to upload scan batches to Cropilot and later download the resulting crop predictions.

The workflow consists of two steps:

1. A folder of uncropped scans, such as TIFF, JPEG, or PNG files, is downscaled and uploaded as JPG files to the Cropilot processing queue.
2. After processing and optional review in the Cropilot editor, the script downloads the crop predictions and applies them to the original images, preserving the original image metadata.

This makes it possible to use lightweight preview images for AI processing and manual review, while producing high-quality cropped outputs from the original source files.

### How to run

#### 1. Install dependencies

```bash
pip install -r requirements-uploader.txt
```

#### 2. Upload a folder of images

The script outputs a link to the Cropilot editor, where the predictions will become available after processing. You can review and adjust the crop boxes before downloading the final crop instructions.

```bash
python3 uploader.py upload --api-key <GROUP_API_KEY> --input-folder sample_input
```

Options:

```text
-h, --help                  show this help message and exit
--api-key API_KEY           API key for authentication within the given group.
                            You can obtain it from the group settings in the web app.
--api-url API_URL           Base URL of the Cropilot API.
                            Defaults to https://app.cropilot.cz
--input-folder INPUT_FOLDER
                            Input folder path containing images to process.
--crop-model CROP_MODEL
                            Model name to use for position prediction.
--rotation-model ROTATION_MODEL
                            Model name to use for angle prediction.
--name NAME                 Custom title name.
                            Defaults to the input folder name.
```

#### 3. Download predictions and crop the original images

After reviewing and saving the predictions in the Cropilot editor, download the crop instructions and apply them to your original local image folder.

```bash
python3 uploader.py download --api-key <GROUP_API_KEY> --title <TITLE_ID> --input-folder sample_input
```

Options:

```text
-h, --help                  show this help message and exit
--api-key API_KEY           API key for authentication within the given group.
                            You can obtain it from the group settings in the web app.
--api-url API_URL           Base URL of the Cropilot API.
                            Defaults to https://app.cropilot.cz
--input-folder INPUT_FOLDER
                            Input folder path containing the original images.
--output-folder OUTPUT_FOLDER
                            Output folder path where cropped images will be saved.
--title TITLE               Title ID.
```

## `trainer.py`: Fine-tune a custom model

The `trainer.py` script can train and upload a new custom model to Cropilot.

Fine-tuning custom models allows Cropilot to better handle specific or uncommon document types, collections, scanning setups, or crop styles.

### Requirements

#### Create a batch of labeled data

Before running the trainer, create labeled training data in the Cropilot editor. This means uploading one or more titles and correcting their crop boxes so they represent the desired output.

We recommend starting with around 250 labeled boxes in total. After training the first custom model, you can evaluate the results and decide whether to repeat the training loop with more data.

If you train from multiple titles, make sure all titles belong to the same Cropilot group.

#### Have enough resources

Training requires approximately 10 GB of GPU memory.

#### Track progress, optional

Training metadata and metrics can be uploaded to [Comet ML](https://www.comet.com/).

To enable experiment tracking, set the following environment variable:

```bash
COMET_ML_API_KEY=<your-api-key>
```

### How to run

#### 1. Install dependencies

```bash
pip install -r requirements-trainer.txt
```

#### 2. Start training

Provide the group API key, a name for the new model, and a list of title IDs that should be used for training.

```bash
python3 trainer.py --api-key <GROUP_API_KEY> --model-name my-new-model --title-ids title1 title2 title3 --train-position
```

After the script finishes, the new model will be available in the Cropilot UI. You can also use it from `uploader.py` by passing it as the model parameter.

Options:

```text
-h, --help                          show this help message and exit
--api-url API_URL                   Base URL of the API.
                                    Defaults to https://app.cropilot.cz/
--base-model BASE_MODEL             Path to the base YOLO model to fine-tune.
                                    Defaults to base_models/default.pt
--api-key API_KEY                   Group API key for authentication.
--model-name MODEL_NAME             New name of the fine-tuned model.
--title-ids TITLE_IDS [TITLE_IDS ...]
                                    List of title IDs to train on.
--train-rotation                    Train the ResNet rotation model.
--train-position                    Train the YOLO position model.
```

## Tip: Run scripts with `uv`

This repository was built using a package manager uv. You can install all the repository dependencies from `uv.lock`. 

From the repository root, run:

```bash
pip install uv

uv sync
```

Then, run the uploader, as...

```bash
uv run -m cropilot_api_tools.uploader upload --api-key <GROUP_API_KEY> --input-folder sample_input
```

```bash
uv run -m cropilot_api_tools.uploader download --api-key <GROUP_API_KEY> --title <TITLE_ID> --input-folder sample_input
```

And the trainer, as...

```bash
uv run -m cropilot_api_tools.trainer --api-key <GROUP_API_KEY> --model-name my-new-model --title-ids title1 title2 title3 --train-position
```