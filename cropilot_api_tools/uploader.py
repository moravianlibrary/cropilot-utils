import argparse
import json
from io import BytesIO
import os
from PIL import Image, ImageOps
from urllib.parse import urljoin
import cv2
import requests


Image.MAX_IMAGE_PIXELS = 933120000


class CropilotUploader:
    """Automatically crops document pages from images using the Cropilot API.
    Image thumbnails are sent via API for prediction. The returned predictions are then
    used to crop the original high-resolution images.
    """

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.web_url = "https://orezy.test.trinera.cloud"
        self.id = None
        self.coordinates = None
        self.api_key = api_key

        self.authenticate()

    def authenticate(self) -> str:
        """Authenticates with the API and returns an access token.

        Args:
            api_url (str): Base URL of the API.
            username (str): API username.
            password (str): API password.

        Returns:
            str: Access token.
        """
        try:
            response = requests.get(
                url=urljoin(self.api_url, "/groups"),
                headers={"X-API-Key": self.api_key},
            )
            response.raise_for_status()
            group = response.json()[0]
            self.group_id = group["_id"]
        except Exception as e:
            raise Exception("Failed to authenticate. Please check your API key.") from e

        print(f"Successfully authenticated to group: {group['name']}")
    
    def get_settings(self, crop_model: str | None, rotation_model: str | None) -> dict:
        """Returns settings dict for API request based on model choice."""
        if crop_model is None and rotation_model is None:
            print("No models specified, using default group settings.")
            return None
        
        response = requests.get(
            url=urljoin(self.api_url, "models")
        )
        response.raise_for_status()
        models = response.json()

        settings = {}
        if crop_model in models["crop_models"]:
            settings["crop_model"] = crop_model
        else:
            print(f"Warning: crop model '{crop_model}' not found in available models, using 'default'.")
        if rotation_model in models["rotation_models"]:
            settings["rotation_model"] = rotation_model
        else:
            print(f"Warning: rotation model '{rotation_model}' not found in available models, using 'text'.")
        return settings

    def upload_and_compress(self, input_folder: str, crop_model: str | None, rotation_model: str | None, name: str):
        """Uploads and compresses images to the Page Trace API.

        Args:
            input_folder (str): Path to the folder containing input images.
            model (str | None): Type of crop to perform ("inner" or "outer").
            name (str): Title name of the book.
        """
        args = {"settings": self.get_settings(crop_model, rotation_model), "external_id": name}
        response = requests.post(
            url=urljoin(self.api_url, f"create?group_id={self.group_id}"),
            headers={"X-API-Key": self.api_key},
            json=args,
        )
        response.raise_for_status()
        self.id = response.json()["id"]

        images = sorted(os.listdir(input_folder))
        print(f"Uploading {len(images)} images...")
        for img in images:
            with open(os.path.join(input_folder, img), "rb") as f:
                try:
                    im = Image.open(f)
                except Exception as e:
                    print(f"Error opening image {img}, skipping. Error: {e}")
                    continue
                im = im.convert("RGB")
                im = ImageOps.exif_transpose(im)
                im.thumbnail((1200, 1200))
                buf = BytesIO()
                # Convert to JPG
                im.save(buf, format="JPEG")
                img = img.rsplit(".", 1)[0] + ".jpg"

                buf.seek(0)
                response = requests.post(
                    url=urljoin(self.api_url, f"{self.id}/upload-scan"),
                    headers={"X-API-Key": self.api_key},
                    files={"scan_data": (img, buf, "image/jpeg")},
                )

                response.raise_for_status()

        print(f"Created entry with name '{name}' and ID '{self.id}'.")

    def process(self):
        """Queues the uploaded images for ML processing and waits for completion.
        Returns URL to view results.
        """
        print("Predicting page crop coordinates...")
        response = requests.post(
            url=urljoin(self.api_url, f"{self.id}/process"),
            headers={"X-API-Key": self.api_key},
        )
        response.raise_for_status()

        state = requests.get(
            url=urljoin(self.api_url, f"{self.id}/status"),
            headers={"X-API-Key": self.api_key},
        )
        print(f"Current status: {state.text}")
        response.raise_for_status()

        print(
            "Processing! Results will be available at",
            urljoin(self.web_url, f"book/{self.id}"),
        )

    def download_results(self, output_folder: str):
        """Downloads the predicted crop coordinates from the API, saves them to disk.

        Args:
            output_folder (str): Path to the folder to save the coordinates JSON.
        """
        response = requests.get(
            url=urljoin(self.api_url, f"{self.id}/scans"),
            headers={"X-API-Key": self.api_key},
        )
        response.raise_for_status()

        self.coordinates = response.json()["scans"]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(os.path.join(output_folder, "coordinates.json"), "w") as f:
            f.write(json.dumps(self.coordinates, indent=4))

    def crop_documents(self, input_folder: str, output_folder: str):
        """Crops the original high-resolution images using the downloaded coordinates.
        Args:
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to the folder to save cropped images.
        """
        images = sorted(os.listdir(input_folder))
        print(f"Cropping {len(images)} images...")

        for img_name, coordinate in zip(images, self.coordinates):
            im = cv2.imread(os.path.join(input_folder, img_name))
            h, w = im.shape[0], im.shape[1]
            for i, page in enumerate(coordinate["pages"]):
                output_image = im.copy()

                xc = page["xc"] * w
                yc = page["yc"] * h
                ww = page["width"] * w
                hh = page["height"] * h
                # rotate image around center
                M = cv2.getRotationMatrix2D((xc, yc), page["angle"], 1.0)
                output_image = cv2.warpAffine(
                    output_image, M, (w, h), flags=cv2.INTER_CUBIC
                )
                # crop image
                output_image = output_image[
                    int(yc - hh / 2) : int(yc + hh / 2),
                    int(xc - ww / 2) : int(xc + ww / 2),
                ]

                # get properties of the original image
                with Image.open(os.path.join(input_folder, img_name)) as orig_img:
                    save_kwargs = orig_img.info.copy()
                    img_extension = orig_img.format.lower()

                # rebuild output_path with preferred extension
                output_path = os.path.join(
                    output_folder, f"{os.path.splitext(img_name)[0]}_page{i + 1}.{img_extension}"
                )

                # convert cv2 BGR/BGRA -> PIL RGB/RGBA
                if output_image.ndim == 2:
                    pil_img = Image.fromarray(output_image)
                else:
                    if output_image.shape[2] == 3:
                        pil_img = Image.fromarray(
                            cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                        )
                    elif output_image.shape[2] == 4:
                        pil_img = Image.fromarray(
                            cv2.cvtColor(output_image, cv2.COLOR_BGRA2RGBA)
                        )
                    else:
                        pil_img = Image.fromarray(output_image)

                pil_img.save(output_path, format=img_extension.upper(), **save_kwargs)

        print(f"Success! Cropped images saved to {output_folder}")

    def upload_job(self, input_folder: str, crop_model: str, rotation_model: str, name: str):
        """Runs the full Page Trace process: upload, process, download, and crop.

        Args:
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to the folder to save cropped images.
            model (str): Type of crop to perform ("inner" or "outer").
            name (str): Title name of the book.
        """
        try:
            self.upload_and_compress(input_folder, crop_model, rotation_model, name)
            self.process()
        except Exception as e:
            if self.id:
                requests.delete(
                    url=urljoin(self.api_url, f"{self.id}"),
                    headers={"X-API-Key": self.api_key},
                )
                print(f"Deleted job {self.id} due to error.")
            raise e

    def download_job(self, title_id: str, input_folder: str, output_folder: str):
        """Downloads results and crops documents.
        Args:
            title_id (str): Title ID for the job.
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to the folder to save cropped images.
        """
        self.id = title_id
        self.download_results(output_folder)
        self.crop_documents(input_folder, output_folder)

        print(f"Job completed successfully, scans saved to {output_folder}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="page_tracer.py")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--api-key",
        type=str,
        help="API key for authentication within given group, obtain from group settings in the web app",
    )
    common.add_argument(
        "--api-url",
        type=str,
        required=False,
        help="Base URL of the Page Trace API, defaults to https://api.ai-orezy.trinera.cloud",
        default="https://api.ai-orezy.trinera.cloud",
    )
    common.add_argument(
        "--input-folder",
        type=str,
        help="Input folder path (containing images to process)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    upload_parser = subparsers.add_parser(
        "upload",
        parents=[common],
        help="Uploads downsized images and schedules job for predictions, outputs link where results will be available.",
    )
    upload_parser.add_argument(
        "--crop-model",
        type=str,
        required=False,
        help="Model name to use for position prediction",
        default=None,
    )
    upload_parser.add_argument(
        "--rotation-model",
        type=str,
        required=False,
        help="Model name to use for angle prediction",
        default=None,
    )
    upload_parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="Custom title name, defaults to input folder name",
    )

    download_parser = subparsers.add_parser(
        "download",
        parents=[common],
        help="Downloads predictions with user updates and crops the original images.",
    )
    download_parser.add_argument(
        "--output-folder",
        type=str,
        required=False,
        help="Output folder path (to save cropped images)",
        default="output",
    )
    download_parser.add_argument(
        "--title",
        type=str,
        required=True,
        help="Title ID",
    )
    args = parser.parse_args()

    print(f"Starting Cropilot Uploader {args.command} for folder {args.input_folder}...")

    tracer = CropilotUploader(api_url=args.api_url, api_key=args.api_key)

    if args.command == "upload":
        if not args.name:
            args.name = os.path.basename(args.input_folder)
        tracer.upload_job(
            input_folder=args.input_folder,
            crop_model=args.crop_model,
            rotation_model=args.rotation_model,
            name=args.name,
        )
    elif args.command == "download":
        tracer.download_job(
            title_id=args.title,
            input_folder=args.input_folder,
            output_folder=args.output_folder,
        )
