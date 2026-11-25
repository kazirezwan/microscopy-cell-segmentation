"""A small U-Net for binary cell segmentation."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())


class UNet(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.e1, self.e2, self.e3 = block(1, base), block(base, 2 * base), block(2 * base, 4 * base)
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.d2 = block(4 * base, 2 * base)
        self.u1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.d1 = block(2 * base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


def train(images: np.ndarray, masks: np.ndarray, epochs: int = 15, batch: int = 16, seed: int = 0):
    torch.manual_seed(seed)
    model = UNet()
    X = torch.tensor(images[:, None]); Y = torch.tensor(masks[:, None].astype(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    bce = nn.BCEWithLogitsLoss()
    hist = []
    for _ in range(epochs):
        model.train()
        perm, total = torch.randperm(len(X)), 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            x, y = X[idx], Y[idx]
            if torch.rand(1) < 0.5:   # flips as augmentation
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1) < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        hist.append(total / len(X))
    return model.eval(), hist


@torch.no_grad()
def predict(model, images: np.ndarray) -> np.ndarray:
    return (torch.sigmoid(model(torch.tensor(images[:, None]))) > 0.5).numpy()[:, 0]
