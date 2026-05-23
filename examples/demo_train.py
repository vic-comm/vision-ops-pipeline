"""
Demo training script: creates synthetic dataset, trains a ResNet50 for 2 epochs,
saves a checkpoint compatible with the API's `InferenceEngine` and a threshold.

Usage:
  python examples/demo_train.py

Outputs:
  artifacts/efficientnet_b3_model.pth
  artifacts/efficientnet_b3_threshold.pkl
"""
from pathlib import Path
import random
try:
    import joblib
except Exception:
    import pickle as joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np


class SyntheticCastingDataset(Dataset):
    def __init__(self, n_per_class=100, image_size=224, transform=None):
        self.samples = []
        self.transform = transform
        self.image_size = image_size

        for _ in range(n_per_class):
            self.samples.append((self._make_image(defect=False), 0))
            self.samples.append((self._make_image(defect=True), 1))

    def _make_image(self, defect: bool):
        img = Image.new("RGB", (self.image_size, self.image_size), (200, 200, 200))
        if defect:
            draw = ImageDraw.Draw(img)
            # draw a dark circle as a synthetic defect
            cx = random.randint(32, self.image_size - 32)
            cy = random.randint(32, self.image_size - 32)
            r = random.randint(8, 20)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(20, 20, 20))
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(dropout=0.3, num_classes=2, device=None):
    backbone = models.resnet50(weights=None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    if device:
        backbone = backbone.to(device)
    return backbone


def main():
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = SyntheticCastingDataset(n_per_class=80, image_size=224, transform=transform)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = build_model(dropout=0.3, num_classes=2, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    for epoch in range(2):
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        print(f"Epoch {epoch+1} loss={total_loss/total:.4f} acc={correct/total:.4f}")

    # Save checkpoint in the format expected by api_service.InferenceEngine
    checkpoint = {
        "model_type": "resnet50",
        "params": {"dropout": 0.3},
        "model_state": model.cpu().state_dict(),
    }
    torch.save(checkpoint, out_dir / "efficientnet_b3_model.pth")

    # Save a demo threshold
    joblib.dump(0.5, out_dir / "efficientnet_b3_threshold.pkl")

    print("Saved demo model and threshold to artifacts/")


if __name__ == "__main__":
    main()
