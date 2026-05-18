import os
from dataclasses import dataclass
from typing import Tuple

import comet_ml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


class DegreeHead(nn.Module):
    """Features -> degrees in [-angle_max, angle_max] via tanh bound."""

    def __init__(self, in_features: int, angle_max: float = 10.0):
        super().__init__()
        self.angle_max = float(angle_max)
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        raw = self.net(x)
        return torch.tanh(raw) * self.angle_max


class AngleDegModel(nn.Module):
    def __init__(self, angle_max: float = 10.0):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1")
        in_feats = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base
        self.head = DegreeHead(in_feats, angle_max)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)  # (B,1)

    def train_model(
        self, train_loader, val_loader, config: "TrainConfig", device: torch.device
    ):
        experiment = comet_ml.get_global_experiment()
        optimizer = optim.AdamW(
            self.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        criterion = nn.HuberLoss(delta=1.0)
        scaler = torch.amp.GradScaler()

        best_mae = float("inf")
        start_epoch = 0
        if config.resume:
            start_epoch, best_mae = load_checkpoint(
                config.resume, self, optimizer, map_location=device
            )
            print(
                f"Resumed from {config.resume}, starting at epoch {start_epoch}, best MAE {best_mae:.4f}"
            )

        for epoch in range(start_epoch, config.epochs):
            train_loss = train_one_epoch(
                self, train_loader, optimizer, scaler, device, criterion
            )
            val_loss, val_mae = evaluate(self, val_loader, device, criterion)

            print(
                f"Epoch [{epoch + 1}/{config.epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val MAE: {val_mae:.4f}"
            )

            experiment.log_metric(
                "train_loss", train_loss, step=epoch
            )
            experiment.log_metric(
                "val_loss", val_loss, step=epoch
            )
            experiment.log_metric("val_mae", val_mae, step=epoch)

            # Save best model
            if val_mae < best_mae:
                best_mae = val_mae
                best_save_path = os.path.join(config.out_dir, "best_model.pth")
                save_checkpoint(best_save_path, self, optimizer, epoch + 1, best_mae)
                print(f"Saved best model to {best_save_path} with MAE {best_mae:.4f}")

            # Save every 10 epochs
            if epoch % 10 == 0:
                save_path = os.path.join(config.out_dir, f"model_epoch_{epoch + 1}.pth")
                save_checkpoint(save_path, self, optimizer, epoch + 1, best_mae)
                print(f"Saved intermediate model to {save_path}")

            # Save last epoch model
            last_save_path = os.path.join(config.out_dir, "last_model.pth")
            save_checkpoint(last_save_path, self, optimizer, epoch + 1, best_mae)

            experiment.log_model(
                "best_model",
                best_save_path,
                metadata={"epoch": epoch + 1, "val_mae": best_mae},
            )


    def predict_angles(self, loader: DataLoader, device: torch.device) -> np.ndarray:
        self.eval()
        preds = []
        trues = []
        all_images = []
        with torch.no_grad():
            for imgs, targets, true_deg in tqdm(loader, desc="Predict", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                outputs = self(imgs)
                preds.append(outputs.detach().cpu().numpy().reshape(-1))
                trues.append(true_deg)
                all_images.append(imgs.detach().cpu().numpy())
        print(
            len(np.concatenate(preds)),
            len(np.concatenate(trues)),
            len(np.concatenate(all_images)),
        )
        return np.concatenate(all_images), np.concatenate(preds), np.concatenate(trues)


# -------------------------------
#  Training utils
# -------------------------------
@dataclass
class TrainConfig:
    out_dir: str = "rotate_finetune_model"
    image_size: int = 640
    batch_size: int = 32
    epochs: int = 150
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    amp: bool = True
    resume: str = None
    angle_max: float = 10.0
    aug_rotate_prob: float = 0.95


def mae_deg(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.abs(pred - true).mean())


def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    total_loss = 0.0
    for imgs, targets, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)  # (B,1) degrees

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(imgs)  # (B,1) in [-angle_max, angle_max]
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    preds, trues, all_images = [], [], []
    for imgs, targets, true_deg in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * imgs.size(0)

        preds.append(outputs.detach().cpu().numpy().reshape(-1))
        trues.append(np.asarray(true_deg, dtype=np.float32))
        all_images.append(imgs.permute(0, 2, 3, 1).cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    all_images = np.concatenate(all_images)

    # Log images to Comet.ml
    for i in range(min(8, len(preds))):
        img = all_images[i]
        img = (img * 0.25) + 0.5  # unnormalize
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # Log images to Comet.ml
        comet_ml.get_global_experiment().log_image(
            Image.fromarray(img),
            name=f"val_img_{i}_pred_{preds[i]:.2f}_true_{trues[i]:.2f}",
            image_format="png",
            overwrite=False,
            step=None,
            metadata={
                "pred_angle": float(preds[i]),
                "true_angle": float(trues[i]),
            },
        )
    return total_loss / len(loader.dataset), mae_deg(preds, trues)


def save_checkpoint(path, model, optimizer, epoch, best_mae):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_mae": best_mae,
        },
        path,
    )


def save_checkpoint_compressed(path, model):
    torch.save({"model": model.state_dict()}, path)
    

def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_mae", float("inf"))
