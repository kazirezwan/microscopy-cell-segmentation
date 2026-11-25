"""Synthetic fluorescence microscopy images with ground-truth masks.

Each image contains elliptical cells of varying size, brightness and
orientation (some touching), on a background with uneven illumination,
optical blur and shot noise. Difficulty controls noise and illumination.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse


def make_image(size: int = 96, n_cells=(6, 16), difficulty: float = 0.5, rng=None):
    rng = rng or np.random.default_rng()
    labels = np.zeros((size, size), dtype=np.int32)
    signal = np.zeros((size, size), dtype=np.float32)
    k = 0
    for _ in range(rng.integers(*n_cells)):
        r, c = rng.uniform(6, size - 6, 2)
        a, b = rng.uniform(3.5, 8.0), rng.uniform(3.0, 6.5)
        rr, cc = ellipse(r, c, a, b, shape=labels.shape, rotation=rng.uniform(0, np.pi))
        if (labels[rr, cc] > 0).mean() > 0.3:   # avoid heavy overlap
            continue
        k += 1
        labels[rr, cc] = k
        signal[rr, cc] = rng.uniform(0.35, 1.0)
    yy, xx = np.mgrid[0:size, 0:size] / size
    gx, gy = rng.uniform(-1, 1, 2)
    illumination = difficulty * 0.45 * (0.5 + 0.5 * (gx * xx + gy * yy))
    img = gaussian_filter(signal, 1.0) + illumination
    photons = 40 + 160 * (1 - difficulty)
    img = rng.poisson(np.clip(img, 0, None) * photons) / photons
    img = img + rng.normal(0, 0.03 + 0.07 * difficulty, img.shape)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.astype(np.float32), labels


def make_dataset(n: int, size: int = 96, difficulty=(0.2, 0.9), seed: int = 0):
    rng = np.random.default_rng(seed)
    imgs, labs, diffs = [], [], []
    for _ in range(n):
        d = rng.uniform(*difficulty)
        i, l = make_image(size, difficulty=d, rng=rng)
        imgs.append(i); labs.append(l); diffs.append(d)
    return np.stack(imgs), np.stack(labs), np.array(diffs)
