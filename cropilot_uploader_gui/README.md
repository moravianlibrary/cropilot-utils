# Cropilot Uploader for macOS

A native SwiftUI macOS app for the existing `cropilot_api_tools/uploader.py` workflow.

The app wraps the same two-step process used by the command-line script:

1. Upload a folder of scans to Cropilot and receive a link to the web editor.
2. After review, download the saved crop instructions and crop the original images locally.

## Build the app

To produce a distributable `.app`, run:

```bash
./cropilot_uploader_gui/build_macos_app.sh
```

The app bundle will be created at:

```text
cropilot_uploader_gui/dist/Cropilot Uploader.app
```

You can zip that `.app` and send it to non-terminal users.

## Run during development

```bash
cd cropilot_uploader_gui
swift run
```

## User flow

- Enter the Cropilot group API key.
- Load available crop and rotation models, or leave both as Default.
- Choose the input folder of original scans.
- Optionally set a title name. If left blank, the input folder name is used.
- Upload the folder. The app will keep checking Cropilot processing status.
- Open the editor link once the app reports that processing is ready.
- After reviewing and saving crops in Cropilot, switch to Download.
- The generated title ID is filled in automatically after upload.
- Choose an output folder.
- Download crop instructions and crop the original images locally.

## Notes

- API URL defaults to `https://api.cropilot.trinera.cloud`, matching the current uploader script.
- Input folder: choose the original folder of scans for both upload and download.
- Output folder: cropped images are written here during download.
- The app is unsigned. On first launch, macOS may require control-clicking the app and choosing Open, or signing/notarizing it before distribution.
