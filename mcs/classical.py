"""Classical baseline: background correction, Otsu threshold, watershed split."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import disk, remove_small_objects, white_tophat
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def segment(img: np.ndarray, min_size: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary mask, instance labels)."""
    corrected = white_tophat(gaussian(img, 1.0), disk(10))  # removes uneven background
    mask = corrected > threshold_otsu(corrected)
    mask = remove_small_objects(ndi.binary_fill_holes(mask), max_size=min_size - 1)
    return mask, split_instances(mask)


def split_instances(mask: np.ndarray, min_distance: int = 4) -> np.ndarray:
    dist = ndi.distance_transform_edt(mask)
    peaks = peak_local_max(dist, min_distance=min_distance, labels=ndi.label(mask)[0])
    markers = np.zeros_like(mask, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-dist, markers, mask=mask)
