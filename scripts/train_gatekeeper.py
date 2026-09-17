"""
Tier-1 Anatomical Gatekeeper Training & Evaluation Script
===========================================================
Trains a lightweight binary classifier (MobileNetV3-Small) to distinguish
legitimate Frontal Chest Radiographs (Class 1) from Non-Chest-X-rays / OOD images
(Class 0, e.g. vehicles, animals, scenery, natural photographs).

Data Source Architecture:
- Class 1 (Positive): Chest X-ray images from data/samples/ or data/raw/images/
- Class 0 (Negative): Natural objects / everyday scenes (downloaded via torchvision CIFAR-10
  or custom negative folder), converted to both grayscale and RGB with contrast jittering
  to ensure the model learns thoracic anatomy rather than mere monochrome color.

Reversibility & Safety:
- This script saves to checkpoints/gatekeeper/gatekeeper_mobilenet_v3.pth
- It NEVER modifies or overwrites the primary diagnostic DenseNet-121 model (checkpoints/phase6/final/best.pth).
- Controlled in app/config.py via ENABLE_ANATOMICAL_GATEKEEPER.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GatekeeperDataset(Dataset):
    """
    Balanced dataset combining positive Chest X-rays and negative Non-X-rays.
    Negative images undergo random grayscale conversion to teach the classifier
    anatomical thoracic patterns rather than simple color absence.
    """
    def __init__(self, items: List[Tuple[Path, int]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


def get_transforms(image_size: int = 224):
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def prepare_data(
    xray_dir: Path,
    non_xray_dir: Optional[Path] = None,
    val_split: float = 0.20
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    positive_files = list(xray_dir.glob("*.png")) + list(xray_dir.glob("*.jpg")) + list(xray_dir.glob("*.jpeg"))
    if not positive_files:
        raise FileNotFoundError(f"No positive X-ray images found in {xray_dir}")

    print(f"[+] Found {len(positive_files)} positive chest radiograph candidates in {xray_dir}")

    negative_files = []
    if non_xray_dir and non_xray_dir.exists():
        negative_files = list(non_xray_dir.glob("*.png")) + list(non_xray_dir.glob("*.jpg")) + list(non_xray_dir.glob("*.jpeg"))

    if not negative_files:
        print("[+] Sourcing neatly labeled negative samples via torchvision CIFAR-10...")
        import torchvision.datasets as dset
        cifar_cache = PROJECT_ROOT / "data" / "cache" / "cifar10_negatives"
        cifar_cache.mkdir(parents=True, exist_ok=True)
        
        cifar_ds = dset.CIFAR10(root=str(PROJECT_ROOT / "data" / "cache"), train=False, download=True)
        num_negatives = max(len(positive_files) * 2, 30)
        
        for idx in range(min(num_negatives, len(cifar_ds))):
            c_img, _ = cifar_ds[idx]
            if idx % 2 == 1:
                c_img = c_img.convert("L").convert("RGB")
            out_p = cifar_cache / f"negative_sample_{idx:05d}.png"
            if not out_p.exists():
                c_img.save(out_p)
            negative_files.append(out_p)

    print(f"[*] Found {len(negative_files)} negative (non-xray/OOD) samples")

    pos_items = [(p, 1) for p in positive_files]
    neg_items = [(p, 0) for p in negative_files]

    np.random.seed(42)
    np.random.shuffle(pos_items)
    np.random.shuffle(neg_items)

    pos_val_len = max(1, int(len(pos_items) * val_split))
    neg_val_len = max(1, int(len(neg_items) * val_split))

    train_items = pos_items[pos_val_len:] + neg_items[neg_val_len:]
    val_items = pos_items[:pos_val_len] + neg_items[:neg_val_len]

    np.random.shuffle(train_items)
    np.random.shuffle(val_items)

    print(f"[*] Dataset split: {len(train_items)} train samples, {len(val_items)} validation samples")
    return train_items, val_items


def build_gatekeeper_model() -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, 1)
    return model


def train_gatekeeper(
    xray_dir: Path,
    non_xray_dir: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-4,
    output_checkpoint: Optional[Path] = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Training Tier-1 Gatekeeper on device: {device}")

    train_items, val_items = prepare_data(xray_dir, non_xray_dir)
    train_tf, val_tf = get_transforms()

    train_loader = DataLoader(GatekeeperDataset(train_items, train_tf), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(GatekeeperDataset(val_items, val_tf), batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_gatekeeper_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_val_acc = 0.0
    output_checkpoint = output_checkpoint or (PROJECT_ROOT / "checkpoints" / "gatekeeper" / "gatekeeper_mobilenet_v3.pth")
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            total_train += len(labels)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)

                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                total_val += len(labels)

        train_acc = train_correct / max(total_train, 1)
        val_acc = val_correct / max(total_val, 1)

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss/total_train:.4f} | Train Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss/total_val:.4f} | Val Acc: {val_acc*100:.1f}%"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "architecture": "mobilenet_v3_small",
                "target_class": "frontal_chest_radiograph"
            }, output_checkpoint)
            print(f"  --> Saved new best gatekeeper checkpoint to {output_checkpoint}")

    print(f"\n[+] Gatekeeper Training Complete! Best Validation Accuracy: {best_val_acc*100:.1f}%")
    print(f"[+] Diagnostic model (best.pth) is untouched. Gatekeeper saved to {output_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tier-1 Chest X-ray Gatekeeper Classifier")
    parser.add_argument("--xray-dir", type=Path, default=PROJECT_ROOT / "data" / "samples",
                        help="Directory containing positive chest X-rays")
    parser.add_argument("--non-xray-dir", type=Path, default=None,
                        help="Directory containing negative non-xray images")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()


    train_gatekeeper(
        xray_dir=args.xray_dir,
        non_xray_dir=args.non_xray_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
