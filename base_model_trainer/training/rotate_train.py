import os

import numpy as np
from base_model_trainer.network.rotate_network import AngleDegModel, TrainConfig, load_checkpoint
from base_model_trainer.network.rotate_dataset import PageAngleDataset
from torch.utils.data import DataLoader
import comet_ml
import torch


def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_filepaths(dir: str) -> list[str]:
    files = sorted([os.path.join(dir, f) for f in os.listdir(dir)])

    return [f for f in files if os.path.isfile(f) and not f.startswith("double_")]


def get_bbox_vectors(dir: str) -> list[tuple[float, float, float, float]]:
    filepaths = get_filepaths(dir)

    bbox_vectors = []
    for fp in filepaths:
        with open(fp, "r") as f:
            try:
                line = f.readline().strip()
            except Exception as e:
                print(f"Error reading file {fp}: {e}")
                continue
            coords = list(map(float, line.split(" ")[1:]))

            if not len(coords) == 4:
                coords = [0.5, 0.5, 1.0, 1.0]
                print(f"Warning: file {fp} has invalid bbox format, using default {coords}")

            bbox_vectors.append(tuple(coords))

    return bbox_vectors


def train():
    cfg = TrainConfig()
    device = set_device()
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")

    train_ds = PageAngleDataset(
        image_paths=get_filepaths("datasets/yolo-all-batches-rotate/images/train"),
        image_bboxes=get_bbox_vectors(
            "datasets/yolo-all-batches-rotate/labels/train"
        ),
        is_train=True,
        image_size=cfg.image_size,
        angle_max=cfg.angle_max,
        # use_canny=True,
    )
    val_ds = PageAngleDataset(
        image_paths=get_filepaths("datasets/yolo-all-batches-rotate/images/val"),
        image_bboxes=get_bbox_vectors(
            "datasets/yolo-all-batches-rotate/labels/val"
        ),
        is_train=True,
        image_size=cfg.image_size,
        angle_max=cfg.angle_max,
        # use_canny=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model = AngleDegModel().to(device)
    torch.compile(model)

    comet_ml.login(
        project_name="autorotate_finetune", api_key=os.getenv("COMET_ML_API_KEY")
    )
    comet_ml.Experiment(project_name="autorotate_finetune")

    model.train_model(
        train_loader,
        val_loader,
        cfg,
        device,
    )


def test():
    import streamlit as st

    device = torch.device("mps")

    test_ds = PageAngleDataset(
        image_paths=get_filepaths("datasets/yolo-all-batches-rotate/images/test"),
        image_bboxes=get_bbox_vectors(
            "datasets/yolo-all-batches-rotate/labels/test"
        ),
        is_train=True,
        image_size=640,
        angle_max=10.0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    model = AngleDegModel().to(device)
    checkpoint_path = "rotate_finetune_model/best_model.pth"
    load_checkpoint(checkpoint_path, model, map_location=device)

    imgs, preds, trues = model.predict_angles(test_loader, device)
    mae = np.mean(np.abs(preds - trues))
    st.write(f"Mean Absolute Error on test set: {mae:.4f} degrees")

    cols = [st.columns(4) for _ in range(len(imgs) // 4)]
    idx = 0
    for img, pred, true in zip(imgs, preds, trues):
        # imgs: (B, C, H, W), angles: (B, 1)
        img_np = img.transpose(1, 2, 0)  # → (H, W, C)
        img_np = np.clip(
            (img_np * 0.25) + 0.5, 0, 1
        )  # inverse normalize: std=0.25, mean=0.5

        img_disp = (img_np * 255).astype(np.uint8)
        cols[idx // 4][idx % 4].image(
            img_disp,
            caption=f"True: {true:.2f}°\nPred: {pred:.2f}°",
        )
        idx += 1

        # Optional: stop after a few batches so Streamlit doesn’t overload
        if idx >= 100:
            break


if __name__ == "__main__":
    train()
    # test()
