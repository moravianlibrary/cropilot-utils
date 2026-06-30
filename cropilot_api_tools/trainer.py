import argparse
import hashlib
import io
import logging
import math
import os
import random
from urllib.parse import urljoin

import comet_ml
import requests
import torch
import yaml
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader
from ultralytics import YOLO

from base_model_trainer.network.rotate_dataset import PageAngleDataset
from base_model_trainer.network.rotate_network import (
    AngleDegModel,
    TrainConfig,
    load_checkpoint,
)
from base_model_trainer.training.rotate_train import (
    get_bbox_vectors,
    get_filepaths,
    set_device,
)
from cropilot_api_tools.config import ConfigError, JobConfig, load_jobs, select_jobs

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"
os.environ["COMET_LOG_ARGUMENTS"] = "0"

COMET_PROJECT = "crop-finetune-domain-specific"

# Hardcoded YOLO hyperparameters that are intentionally not configurable.
YOLO_IMGSZ = 640
YOLO_MAX_DET = 2


def page_aabb_px(
    page: dict, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Pixel (cx, cy, w, h) of the axis-aligned bounding box of a page.

    The page is an oriented rectangle (upright size width x height, rotated by
    `angle`); its axis-aligned bounding box in the raw, un-deskewed image is what
    the YOLO position model predicts — matching production, where scans are fed
    to the model without deskewing.
    """
    r = math.radians(page["angle"])
    w_px = page["width"] * img_w
    h_px = page["height"] * img_h
    aabb_w = abs(w_px * math.cos(r)) + abs(h_px * math.sin(r))
    aabb_h = abs(w_px * math.sin(r)) + abs(h_px * math.cos(r))
    return page["xc"] * img_w, page["yc"] * img_h, aabb_w, aabb_h


def double_page_boxes(
    pages: list, img_w: int, img_h: int
) -> list[tuple[float, float, float, float]]:
    """Normalized axis-aligned YOLO boxes for a raw (un-deskewed) spread."""
    boxes = []
    for page in pages:
        cx, cy, w, h = page_aabb_px(page, img_w, img_h)
        boxes.append(
            (
                min(max(cx / img_w, 0.0), 1.0),
                min(max(cy / img_h, 0.0), 1.0),
                min(w / img_w, 1.0),
                min(h / img_h, 1.0),
            )
        )
    return boxes


class CropilotTrainer:
    """Automates fine-tuning of Cropilot models from a resolved JobConfig."""

    def __init__(self, job: JobConfig):
        self.job = job
        self.api_url = job.api_url
        self.api_key = job.api_key
        self.base_model = job.base_model
        self.model_name = job.name

        self.directory = f"finetune_dataset_{self.model_name}"

        # Image basenames the YOLO position model trains on:
        #  - "single_<scan>_p<i>": raw single-page image (one box) — for a
        #    multi-page scan these are the spread cut at the gutter into halves;
        #  - "double_<scan>": the raw full spread (two boxes).
        # The deskewed per-page crops "<scan>_p<i>" are for the rotation model
        # only and are excluded from the YOLO lists.
        self.position_basenames: set[str] = set()

        # Maps scan id -> all image basenames produced for it, so the train/val
        # split can keep every artifact of a scan on the same side (no leakage).
        self.scan_artifacts: dict[str, list[str]] = {}

        # Dataset statistics collected during download, logged to Comet.
        self.stats = {
            "num_scans": 0,
            "num_empty_labels": 0,
            "num_multipage_scans": 0,
            "num_samples": 0,
            "num_train": 0,
            "num_val": 0,
        }

        self.authenticate()

    def authenticate(self) -> None:
        """Authenticates with the API and resolves the group id."""
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

        logger.info(f"Successfully authenticated to group: {group['name']}")

    def prepare_directories(self) -> None:
        """Creates the dataset directory tree, clearing any stale leftover.

        Removing a leftover directory from a previously crashed run first means
        a re-run does not fail on directory creation.
        """
        if os.path.exists(self.directory):
            logger.info(f"Removing stale dataset directory {self.directory}")
            self.cleanup()
        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            os.makedirs(f"{self.directory}/{sub}", exist_ok=True)

    def train_job(self, exp: comet_ml.CometExperiment) -> None:
        """Main training job orchestrator."""
        self.prepare_directories()

        try:
            for title_id in self.job.title_ids:
                self.download_training_data(title_id)
            self.split_train_val()
            self.create_dataset_yaml()
            self.check_dataset()

            # Log dataset statistics once the dataset is fully built.
            exp.log_metrics(self.stats)

            if self.job.train_position:
                results = self.finetune_crop_model()
                self.log_position_metrics(exp, results)
                self.upload_trained_crop_model(exp, results.save_dir)

            if self.job.train_rotation:
                self.finetune_rotation_model()

            logger.info(f"Training job '{self.model_name}' completed successfully.")
        finally:
            self.cleanup()

    def check_dataset(self) -> None:
        """Fails early with a clear message if the built dataset is unusable."""
        if self.stats["num_samples"] == 0:
            raise RuntimeError(
                f"Job '{self.model_name}': no training data was downloaded for "
                f"title_ids {self.job.title_ids}. Check the IDs and API key."
            )
        if self.stats["num_train"] == 0 or self.stats["num_val"] == 0:
            raise RuntimeError(
                f"Job '{self.model_name}': empty train or val split "
                f"({self.stats['num_train']}/{self.stats['num_val']}). "
                "Provide more data or lower val_split."
            )
        if self.job.train_position and not self.position_basenames:
            raise RuntimeError(
                f"Job '{self.model_name}': train_position is set but no position "
                "samples were produced."
            )

    def finetune_rotation_model(self) -> None:
        """Fine-tunes a separate model for rotation (angle) prediction."""
        rot = self.job.rotation
        rotation_base = rot.get("base_model")
        if not rotation_base:
            raise ConfigError(
                f"Job '{self.model_name}' has train_rotation=true but no "
                "rotation.base_model is configured."
            )

        start_epoch, _ = load_checkpoint(
            rotation_base, AngleDegModel(), map_location="cpu"
        )

        # Build TrainConfig from configurable rotation hyperparameters, falling
        # back to TrainConfig defaults for anything not specified.
        cfg_kwargs = {
            "resume": rotation_base,
            "epochs": start_epoch + rot.get("extra_epochs", 50),
        }
        for key in ("batch_size", "image_size", "angle_max", "lr", "weight_decay", "num_workers"):
            if key in rot:
                cfg_kwargs[key] = rot[key]
        cfg = TrainConfig(**cfg_kwargs)
        os.makedirs(cfg.out_dir, exist_ok=True)

        device = set_device()

        train_ds = PageAngleDataset(
            image_paths=get_filepaths(f"{self.directory}/images/train"),
            image_bboxes=get_bbox_vectors(f"{self.directory}/labels/train"),
            is_train=True,
            image_size=cfg.image_size,
            angle_max=cfg.angle_max,
        )
        val_ds = PageAngleDataset(
            image_paths=get_filepaths(f"{self.directory}/images/val"),
            image_bboxes=get_bbox_vectors(f"{self.directory}/labels/val"),
            is_train=True,
            image_size=cfg.image_size,
            angle_max=cfg.angle_max,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        model = AngleDegModel().to(device)
        model.train_model(train_loader, val_loader, cfg, device)

        torch.save({"model": model.state_dict()}, self.model_name + "_rotate.pth")

    def upload_trained_crop_model(self, exp: comet_ml.CometExperiment, save_dir) -> None:
        """Uploads the trained YOLO model to the API and logs it to Comet.

        `save_dir` is the run directory returned by Ultralytics (results.save_dir),
        which points exactly at this run's outputs — more robust than scanning
        and sorting the runs folder by name.
        """
        best_weights = os.path.join(str(save_dir), "weights", "best.pt")
        os.rename(best_weights, f"{self.model_name}.pt")

        # Register the trained weights as a Comet model asset.
        exp.log_model(self.model_name, f"{self.model_name}.pt")

        with open(f"{self.model_name}.pt", "rb") as f:
            response = requests.post(
                url=urljoin(self.api_url, "/models"),
                headers={"X-API-Key": self.api_key},
                files={"file": f},
            )
        response.raise_for_status()
        logger.info(f"Uploaded trained model: {self.model_name}")

    def log_position_metrics(self, exp: comet_ml.CometExperiment, results) -> None:
        """Logs final YOLO detection metrics (mAP) to Comet, if available."""
        try:
            metrics = getattr(results, "results_dict", None) or {}
            # Keep only the meaningful summary metrics with clean names.
            for key, value in metrics.items():
                exp.log_metric(f"yolo/{key}", value)
        except Exception as e:  # never fail the run because of logging
            logger.warning(f"Could not log YOLO metrics: {e}")

    def download_training_data(self, title_id: str) -> None:
        """Downloads images and labels for a given title ID."""
        response = requests.get(
            url=urljoin(self.api_url, f"{title_id}/scans"),
            headers={"X-API-Key": self.api_key},
        )
        response.raise_for_status()

        scans = response.json()["scans"]
        logger.info(f"Downloaded metadata for title {title_id}, found {len(scans)} scans.")
        for scan in scans:
            self.stats["num_scans"] += 1
            self.save_scan_image(title_id, scan)

    def create_dataset_yaml(self) -> None:
        """Creates the dataset.yaml file required by YOLO.

        The position model trains on explicit image lists (train.txt/val.txt)
        rather than whole image folders, so the deskewed per-page rotation crops
        are kept out of the YOLO dataset while still living on disk for the
        rotation model.
        """
        with open(f"{self.directory}/dataset.yaml", "w") as f:
            f.write(f"path: {os.path.abspath(self.directory)}\n")
            f.write("train: train.txt\n")
            f.write("val: val.txt\n")
            f.write("names:\n")
            f.write("    0: page\n")

    def save_scan_image(self, title_id: str, scan: dict) -> None:
        """Downloads the scan image and writes all training artifacts for a scan.

        Position model (raw images, boxes = axis-aligned bbox at full angle,
        matching production where scans are not deskewed):
          * "single_<scan>_p<i>.jpg": a single page. For a multi-page scan the
            spread is cut at the gutter into one image per page (the artificial
            single pages); for a single-page scan it is the whole image.
          * "double_<scan>.jpg": the full raw spread with one box per page.
        Rotation model (only when train_rotation):
          * "<scan>_p<i>.jpg": the deskewed (and, for spreads, masked) per-page
            crop, with a single-box label.
        """
        response = requests.get(
            url=urljoin(self.api_url, f"{title_id}/files?scan_id={scan['_id']}"),
            headers={"X-API-Key": self.api_key},
        )
        response.raise_for_status()

        base = ImageOps.exif_transpose(Image.open(io.BytesIO(response.content)))
        base = base.convert("RGB")
        scan_id = scan["_id"]
        artifacts = []

        # no_prediction negatives: a single raw image with an empty label.
        if "no_prediction" in scan["flags"] and not scan["edited"]:
            self.stats["num_empty_labels"] += 1
            if self.job.train_position:
                name = f"single_{scan_id}_p0"
                base.save(f"{self.directory}/images/{name}.jpg")
                open(f"{self.directory}/labels/{name}.txt", "w").close()
                self.position_basenames.add(name)
                artifacts.append(name)
            self.scan_artifacts[scan_id] = artifacts
            return

        pages = sorted(scan["pages"], key=lambda p: p["xc"])
        if len(pages) > 1:
            self.stats["num_multipage_scans"] += 1

        # Rotation: deskewed per-page crops + single-box labels in crop frame.
        if self.job.train_rotation:
            for i, page in enumerate(pages):
                crop = self.render_page_crop(base, pages, i)
                name = f"{scan_id}_p{i}"
                crop.save(f"{self.directory}/images/{name}.jpg")
                with open(f"{self.directory}/labels/{name}.txt", "w") as f:
                    f.write(
                        f"0 {page['xc']} {page['yc']} {page['width']} {page['height']}\n"
                    )
                artifacts.append(name)

        # Position: raw full spread (+ raw single pages, sampled per spread).
        if self.job.train_position:
            multipage = len(pages) > 1
            # Genuine single-page scans always contribute their single image;
            # synthetic single pages from spreads are sampled by fraction.
            if not multipage or self.emit_singles_for(scan_id):
                singles = self.save_single_pages(base, scan_id, pages)
                self.position_basenames.update(singles)
                artifacts += singles
            if multipage:
                self.save_double_page(base, scan_id, pages)
                self.position_basenames.add(f"double_{scan_id}")
                artifacts.append(f"double_{scan_id}")

        self.scan_artifacts[scan_id] = artifacts

    def emit_singles_for(self, scan_id: str) -> bool:
        """Whether a spread should also yield single-page crops.

        Deterministic per scan id (so re-runs are reproducible without a global
        seed). `position.single_page_fraction` is the fraction of spreads that
        contribute single pages, giving a singles:doubles ratio of 2*fraction:1.
        """
        fraction = self.job.position.get("single_page_fraction", 0.25)
        if fraction >= 1.0:
            return True
        if fraction <= 0.0:
            return False
        bucket = int(hashlib.md5(scan_id.encode()).hexdigest(), 16) % 1000
        return bucket < fraction * 1000

    def render_page_crop(self, base: Image.Image, pages: list, i: int) -> Image.Image:
        """Deskews a single page; masks the neighbouring page on multi-page scans.

        Used only for the rotation model, which needs an upright single page.
        """
        page = pages[i]
        xc = page["xc"] * base.width
        yc = page["yc"] * base.height
        image = base.rotate(page["angle"], center=(xc, yc))

        if len(pages) > 1:
            # mask object outside of width
            w = page["width"] * image.width * 1.5
            left = 0 if i == 0 else (xc - w / 2)
            right = (xc + w / 2) if i == 0 else image.width

            mask = Image.new("L", image.size, 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rectangle([(left, 0), (right, image.height)], fill=255)
            image.putalpha(mask)
            image = Image.composite(
                image, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask
            ).convert("RGB")

        return image.convert("RGB")

    def save_single_pages(
        self, base: Image.Image, scan_id: str, pages: list
    ) -> list[str]:
        """Cuts a raw spread at the gutter into one single-page image per page.

        Each page becomes "single_<scan>_p<i>.jpg" with its axis-aligned box
        (at the page's full angle) clipped to the crop and renormalized. A
        single-page scan yields one image equal to the whole scan.
        """
        w_img, h_img = base.width, base.height

        # Vertical cut lines: image edges plus the midpoint between adjacent
        # page centers (the gutter), so each page lands in its own crop.
        centers = [p["xc"] * w_img for p in pages]
        cuts = (
            [0.0]
            + [(centers[i] + centers[i + 1]) / 2 for i in range(len(pages) - 1)]
            + [float(w_img)]
        )

        names = []
        for i, page in enumerate(pages):
            x0, x1 = int(cuts[i]), int(cuts[i + 1])
            crop_w = x1 - x0
            crop = base.crop((x0, 0, x1, h_img))

            name = f"single_{scan_id}_p{i}"
            crop.save(f"{self.directory}/images/{name}.jpg")

            cx, cy, bw, bh = page_aabb_px(page, w_img, h_img)
            # Clip the box to the crop window, then renormalize to the crop.
            bx0, bx1 = max(cx - bw / 2, x0), min(cx + bw / 2, x1)
            by0, by1 = max(cy - bh / 2, 0), min(cy + bh / 2, h_img)
            ncx = ((bx0 + bx1) / 2 - x0) / crop_w
            ncy = (by0 + by1) / 2 / h_img
            nw = (bx1 - bx0) / crop_w
            nh = (by1 - by0) / h_img
            with open(f"{self.directory}/labels/{name}.txt", "w") as f:
                f.write(f"0 {ncx} {ncy} {nw} {nh}\n")

            names.append(name)
        return names

    def save_double_page(self, base: Image.Image, scan_id: str, pages: list) -> None:
        """Saves the raw (un-deskewed) full spread with one box per page.

        No rotation is applied, so the image matches production input; each box
        is the axis-aligned bounding box of the page at its full angle.
        """
        base.save(f"{self.directory}/images/double_{scan_id}.jpg")
        boxes = double_page_boxes(pages, base.width, base.height)
        with open(f"{self.directory}/labels/double_{scan_id}.txt", "w") as f:
            for xc, yc, w, h in boxes:
                f.write(f"0 {xc} {yc} {w} {h}\n")

    def split_train_val(self) -> None:
        """Splits the dataset into train/val sets, grouped by scan.

        The split is done per scan id (not per image), so every artifact of a
        scan — its page crops and its full double-page spread — lands on the
        same side. This prevents train/val leakage between near-identical
        images derived from the same scan.
        """
        scan_ids = list(self.scan_artifacts.keys())
        random.shuffle(scan_ids)
        n_train = int((1.0 - self.job.val_split) * len(scan_ids))
        train_scans, val_scans = scan_ids[:n_train], scan_ids[n_train:]

        moved = {"train": 0, "val": 0}
        for split, scans in (("train", train_scans), ("val", val_scans)):
            for scan_id in scans:
                for basename in self.scan_artifacts[scan_id]:
                    os.rename(
                        f"{self.directory}/images/{basename}.jpg",
                        f"{self.directory}/images/{split}/{basename}.jpg",
                    )
                    os.rename(
                        f"{self.directory}/labels/{basename}.txt",
                        f"{self.directory}/labels/{split}/{basename}.txt",
                    )
                    moved[split] += 1

        self.stats["num_samples"] = moved["train"] + moved["val"]
        self.stats["num_train"] = moved["train"]
        self.stats["num_val"] = moved["val"]

        logger.info(
            f"Split {len(scan_ids)} scans into {len(train_scans)} train "
            f"and {len(val_scans)} val ({moved['train']}/{moved['val']} images)."
        )

        self.write_yolo_lists()
        logger.info("Finished splitting data into train and val sets.")

    def write_yolo_lists(self) -> None:
        """Writes train.txt/val.txt listing only the position-model images.

        These are the raw single-page crops ("single_*") and full spreads
        ("double_*"). The deskewed per-page rotation crops are excluded here and
        remain on disk only for the rotation model.
        """
        for split in ("train", "val"):
            paths = []
            for basename in self.position_basenames:
                image_path = os.path.join(
                    self.directory, "images", split, f"{basename}.jpg"
                )
                if os.path.exists(image_path):
                    paths.append(os.path.abspath(image_path))
            with open(f"{self.directory}/{split}.txt", "w") as f:
                f.write("\n".join(paths) + "\n")
            logger.info(f"Wrote {len(paths)} entries to {split}.txt (position model).")

    def finetune_crop_model(self):
        """Fine-tunes the YOLO position model and returns the training results."""
        pos = self.job.position
        train_kwargs = dict(
            data=f"{self.directory}/dataset.yaml",
            project=COMET_PROJECT,
            name=self.model_name,
            # Configurable hyperparameters (defaults < job overrides).
            epochs=pos.get("epochs", 500),
            batch=pos.get("batch", 32),
            scale=pos.get("scale", 0.5),
            flipud=pos.get("flipud", 0.5),
            fliplr=pos.get("fliplr", 0.5),
            close_mosaic=pos.get("close_mosaic", 50),
            degrees=pos.get("degrees", 4),
            shear=pos.get("shear", 1.0),
            patience=pos.get("patience", 100),
            # Hardcoded, non-configurable hyperparameters.
            imgsz=YOLO_IMGSZ,
            max_det=YOLO_MAX_DET,
            save_json=True,
            single_cls=True,
        )
        # Optional fine-tuning knobs: only passed when set, so they otherwise
        # fall back to the Ultralytics defaults.
        if "lr0" in pos:
            train_kwargs["lr0"] = pos["lr0"]
        if "freeze" in pos:
            train_kwargs["freeze"] = pos["freeze"]

        model = YOLO(self.base_model)
        return model.train(**train_kwargs)

    def cleanup(self) -> None:
        """Removes the temporary dataset directory."""
        if os.path.exists(self.directory):
            for root, dirs, files in os.walk(self.directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.directory)
            logger.info(f"Removed directory {self.directory}")
        logger.info("Cleanup completed.")


def run_job(job: JobConfig) -> None:
    """Runs a single job inside its own Comet experiment."""
    comet_ml.login(project_name=COMET_PROJECT, api_key=os.getenv("COMET_ML_API_KEY"))
    # One experiment per job, so --all does not merge runs together.
    exp = comet_ml.start(project_name=COMET_PROJECT)
    try:
        exp.set_name(job.name)
        exp.log_parameters(job.comet_parameters())
        exp.add_tags(job.title_ids)
        exp.add_tag(f"group:{job.group}")
        if job.train_position:
            exp.add_tag("train:crop")
        if job.train_rotation:
            exp.add_tag("train:rotate")
        # Attach the resolved job config for full reproducibility.
        exp.log_asset_data(
            yaml.safe_dump(job.comet_parameters(), sort_keys=False),
            name=f"{job.name}.config.yaml",
        )

        trainer = CropilotTrainer(job)
        trainer.train_job(exp)
    finally:
        exp.end()


def print_jobs(jobs: list[JobConfig]) -> None:
    """Prints a human-readable summary of the selected jobs (--list)."""
    for job in jobs:
        targets = []
        if job.train_position:
            targets.append("position")
        if job.train_rotation:
            targets.append("rotation")
        print(f"\n{job.name}  (group={job.group})")
        print(f"  api_url:    {job.api_url}")
        print(f"  base_model: {job.base_model}")
        print(f"  train:      {', '.join(targets)}")
        print(f"  val_split:  {job.val_split}")
        print(f"  title_ids:  {len(job.title_ids)} -> {', '.join(job.title_ids)}")
        print(f"  position:   {job.position}")
        if job.train_rotation:
            print(f"  rotation:   {job.rotation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="trainer.py")
    parser.add_argument(
        "--config", required=True, help="Path to the jobs YAML config file"
    )
    parser.add_argument(
        "--job",
        action="append",
        dest="jobs",
        default=[],
        help="Name of a job to run (repeatable)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run every job in the config file"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the selected jobs with resolved parameters and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the dataset but skip training (validation only)",
    )

    args = parser.parse_args()

    all_jobs = load_jobs(args.config)

    # --list without a selection lists everything; otherwise honor the selection.
    if args.list and not (args.jobs or args.all):
        print_jobs(all_jobs)
        raise SystemExit(0)

    selected = select_jobs(all_jobs, args.jobs, args.all)

    if args.list:
        print_jobs(selected)
        raise SystemExit(0)

    if args.dry_run:
        for job in selected:
            logger.info(f"[dry-run] Building dataset for '{job.name}'...")
            trainer = CropilotTrainer(job)
            try:
                trainer.prepare_directories()
                for title_id in job.title_ids:
                    trainer.download_training_data(title_id)
                trainer.split_train_val()
                trainer.check_dataset()
                logger.info(f"[dry-run] '{job.name}' dataset stats: {trainer.stats}")
            finally:
                trainer.cleanup()
        raise SystemExit(0)

    for job in selected:
        run_job(job)
