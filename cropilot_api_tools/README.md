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

First of all, install python dependencies with:

```bash
pip install -r requirements-uploader.txt
```

Then, download **Exif Tool** (a postprocessing tool used to copy TIFF metadata from uncropped images). Download based on your OS:

Ubuntu / Debian: `sudo apt install libimage-exiftool-perl`

Fedora: `sudo dnf install perl-Image-ExifTool`

MacOS: `brew install exiftool`

Windows:
1. Download the **Windows Executable** (`exiftool-XX_64.zip`) from https://exiftool.org/index.html and unzip it.
2. Rename `exiftool(-k).exe` to `exiftool.exe`.
3. Keep the `exiftool_files` folder next to `exiftool.exe` — the current ExifTool build needs it to run.
4. Put `exiftool.exe` (together with its `exiftool_files` folder) in the same folder as `uploader.py`.


#### 2. Upload a folder of images

The script outputs a link to the Cropilot editor, where the predictions will become available after processing. You can review and adjust the crop boxes before downloading the final crop instructions. From the folder where your `uploader.py` exists, execute:

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

#### 2. Describe your jobs in a config file

The trainer is driven by a single YAML config file that describes every
fine-tuning job. Copy the provided example and fill in your real API keys:

```bash
cp jobs.example.yaml jobs.yaml
```

> **Note:** `jobs.yaml` contains group API keys, so it is git-ignored. Keep your
> keys out of version control and never paste them on the command line.

The file has three sections. Values are resolved with the precedence
`defaults < group < job`, so each job only needs to state what differs.

```yaml
# Inherited by every job unless overridden.
defaults:
  api_url: https://api.cropilot.trinera.cloud/
  base_model: base_models/default.pt
  val_split: 0.2
  position:            # YOLO position hyperparameters (override per job)
    epochs: 500
    batch: 32
    # ... scale, flipud, fliplr, close_mosaic, degrees, shear
  rotation:            # ResNet rotation hyperparameters
    base_model: base_models/rotate-300e-best.pth
    extra_epochs: 50

# Named groups holding the API key (and optionally a custom api_url).
groups:
  group-name:
    api_key: <GROUP_API_KEY>

# The models to train.
jobs:
  - name: my-new-model
    group: group-name                 # references a group above
    base_model: base_models/default.pt
    train_position: true       # both flags must be stated explicitly
    train_rotation: false
    position:
      epochs: 600              # overrides the default just for this job
    title_ids:
      - title1
      - title2
      - title3
```

Notes:

- `imgsz` and `max_det` are intentionally **not** configurable and stay
  hardcoded in `trainer.py`.
- The position block also accepts `patience` (early-stopping patience) and the
  optional fine-tuning knobs `lr0` (initial learning rate) and `freeze` (number
  of layers to freeze). `lr0`/`freeze` are only passed to YOLO when present,
  otherwise the Ultralytics defaults apply.
- `train_position` and `train_rotation` have no default — set them on every job.
- The position model trains on raw, un-deskewed images that match production:
  the full spread with one box per page ("double_*"), plus raw single-page
  images obtained by cutting each spread at the gutter ("single_*"). The
  rotation model uses the deskewed per-page crops. The train/val split is done
  per scan, so all artifacts of one scan stay on the same side (no leakage).
- Each job runs in its own Comet.ml experiment (so `--all` does not merge runs),
  logging the resolved hyperparameters, dataset statistics, result metrics, and
  the trained model and config as assets.

See `jobs.example.yaml` for a complete, working example.

#### 3. Start training

Select jobs by name (repeatable) or run them all:

```bash
python3 -m cropilot_api_tools.trainer --config jobs.yaml --job my-new-model
```

After the script finishes, the new model will be available in the Cropilot UI. You can also use it from `uploader.py` by passing it as the model parameter.

Options:

```text
-h, --help          show this help message and exit
--config CONFIG     Path to the jobs YAML config file (required).
--job JOB           Name of a job to run. Repeat to run several jobs.
--all               Run every job defined in the config file.
--list              List the selected jobs with resolved parameters and exit.
--dry-run           Build the dataset but skip training (validation only).
```

Examples:

```bash
# Run several jobs
... --config jobs.yaml --job my-new-model --job another-model

# Run everything
... --config jobs.yaml --all

# Inspect what would be trained, without training
... --config jobs.yaml --all --list

# Verify the config and dataset download without training
... --config jobs.yaml --job my-new-model --dry-run
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
uv run -m --env-file cropilot_api_tools/.env cropilot_api_tools.trainer --config cropilot_api_tools/jobs.yaml --job my-new-model
```