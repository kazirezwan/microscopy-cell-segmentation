"""Load real microscopy data from two folders of matching file names.

    images/  img_001.png  img_002.png ...
    masks/   img_001.png  img_002.png ...   (0 = background, >0 = cell / instance id)

Works with e.g. the BBBC or Data Science Bowl 2018 nuclei data after
converting each image's masks to a single label image.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize


def load_folder(images_dir: str | Path, masks_dir: str | Path, size: int | None = 96):
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    imgs, labs = [], []
    for p in sorted(images_dir.iterdir()):
        m = masks_dir / p.name
        if not m.exists():
            continue
        img = io.imread(p).astype(np.float32)
        if img.ndim == 3:
            img = rgb2gray(img[..., :3])
        lab = io.imread(m)
        if lab.ndim == 3:
            lab = lab[..., 0]
        if size:
            img = resize(img, (size, size), anti_aliasing=True)
            lab = resize(lab, (size, size), order=0, preserve_range=True, anti_aliasing=False)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        imgs.append(img.astype(np.float32)); labs.append(lab.astype(np.int32))
    if not imgs:
        raise FileNotFoundError(f"no matching image/mask pairs in {images_dir} and {masks_dir}")
    return np.stack(imgs), np.stack(labs)
