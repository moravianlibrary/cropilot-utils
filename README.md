# Cropilot utils

This repository contains supporting tools and machine learning code for [Cropilot](https://github.com/moravianlibrary/cropilot), an AI-powered application for automatically cropping scanned documents.

Cropilot utils provides the models and scripts used to detect pages in scanned books, newspapers, periodicals, and other printed media. These components support the main Cropilot application by predicting page locations, rotation angles, and crop instructions that can later be reviewed in the Cropilot editor or applied automatically.

## Models

The repository contains the training code for the computer vision models used by Cropilot. If you want to train a model from scratch, see the model training tutorial:

[Train a Cropilot model from scratch](./base_model_trainer/README.md)

Cropilot currently uses two models to generate page predictions.

### Fine-tuned YOLO model

Cropilot uses a fine-tuned YOLO model based on [YOLO11s](https://docs.ultralytics.com/models/yolo11/) to detect the number and position of pages in each scan.

### RotateNET

RotateNET is a ResNet-based model that predicts the rotation angle of each detected page so the final crop can be properly aligned.

## Cropilot Tools

The `cropilot_api_tools` directory contains utility scripts for working with the Cropilot API. These scripts are intended for larger batch workflows that go beyond the simple upload flow available in the web UI.

Cropilot Tools can be used to:

- Upload scan batches to the Cropilot editor.
- Download Cropilot-generated crop instructions.
- Apply crop instructions to original image files.
- Fine-tune custom models for your own document datasets.

See the Cropilot Tools documentation for setup and usage instructions:

[Cropilot Tools Guide](./cropilot_api_tools/README.md)
