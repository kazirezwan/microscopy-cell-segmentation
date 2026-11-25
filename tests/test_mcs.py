import numpy as np
import torch

from mcs import classical, unet
from mcs.metrics import count_error, dice, iou
from mcs.synth import make_dataset, make_image


def test_synthetic_image_and_labels():
    img, lab = make_image(64, rng=np.random.default_rng(0))
    assert img.shape == lab.shape == (64, 64)
    assert 0 <= img.min() and img.max() <= 1
    assert lab.max() >= 1


def test_metrics():
    a = np.zeros((10, 10), bool); a[:5] = True
    assert dice(a, a) == 1.0 and iou(a, a) == 1.0
    assert dice(a, ~a) == 0.0
    lab = np.zeros((10, 10), int); lab[0, 0] = 1; lab[5, 5] = 2
    assert count_error(lab, lab) == 0


def test_classical_finds_cells_on_easy_image():
    img, lab = make_image(96, difficulty=0.1, rng=np.random.default_rng(1))
    mask, inst = classical.segment(img)
    assert dice(mask, lab > 0) > 0.6


def test_unet_shape_and_short_training():
    X, L, _ = make_dataset(16, size=32, seed=0)
    m = unet.UNet(base=8)
    assert m(torch.tensor(X[:2, None])).shape == (2, 1, 32, 32)
    model, hist = unet.train(X, L > 0, epochs=3, batch=8)
    assert hist[-1] < hist[0]


def test_load_folder(tmp_path):
    from skimage import io
    from mcs.io import load_folder
    (tmp_path / "images").mkdir(); (tmp_path / "masks").mkdir()
    for k in range(3):
        img, lab = make_image(64, rng=np.random.default_rng(k))
        io.imsave(tmp_path / "images" / f"{k}.png", (img * 255).astype(np.uint8), check_contrast=False)
        io.imsave(tmp_path / "masks" / f"{k}.png", lab.astype(np.uint8), check_contrast=False)
    X, L = load_folder(tmp_path / "images", tmp_path / "masks", size=32)
    assert X.shape == L.shape == (3, 32, 32)
