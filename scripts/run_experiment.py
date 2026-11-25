"""Classical pipeline vs U-Net on synthetic fluorescence images.

    python scripts/run_experiment.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mcs import classical, unet  # noqa: E402
from mcs.metrics import count_error, dice, iou  # noqa: E402
from mcs.synth import make_dataset  # noqa: E402


def evaluate(pred_masks, pred_labels, labs, diffs):
    res = {}
    for name, sel in [("easy (difficulty < 0.55)", diffs < 0.55), ("hard (difficulty >= 0.55)", diffs >= 0.55),
                      ("all", np.ones_like(diffs, bool))]:
        idx = np.flatnonzero(sel)
        res[name] = dict(
            n=int(len(idx)),
            dice=round(float(np.mean([dice(pred_masks[i], labs[i] > 0) for i in idx])), 4),
            iou=round(float(np.mean([iou(pred_masks[i], labs[i] > 0) for i in idx])), 4),
            mean_abs_count_error=round(float(np.mean([abs(count_error(pred_labels[i], labs[i])) for i in idx])), 3),
        )
    return res


def main():
    t0 = time.time()
    X_tr, L_tr, _ = make_dataset(400, seed=1)
    X_te, L_te, D_te = make_dataset(150, seed=2)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    cl = [classical.segment(x) for x in X_te]
    cl_masks, cl_labels = np.stack([m for m, _ in cl]), np.stack([l for _, l in cl])

    model, hist = unet.train(X_tr, L_tr > 0, epochs=15)
    un_masks = unet.predict(model, X_te)
    un_labels = np.stack([classical.split_instances(m) for m in un_masks])

    results = {"classical (top-hat + Otsu + watershed)": evaluate(cl_masks, cl_labels, L_te, D_te),
               "U-Net (trained on 400 synthetic images)": evaluate(un_masks, un_labels, L_te, D_te),
               "setup": dict(train_images=400, test_images=150, size=96,
                             mean_cells_per_test_image=round(float(np.mean([len(np.unique(l)) - 1 for l in L_te])), 2),
                             runtime_s=round(time.time() - t0, 1))}
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    hard = np.argsort(-D_te)[:3]
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    for r, i in enumerate(hard):
        for c, (img, title) in enumerate([(X_te[i], f"input (difficulty {D_te[i]:.2f})"),
                                          (L_te[i] > 0, "ground truth"),
                                          (cl_masks[i], f"classical, Dice {dice(cl_masks[i], L_te[i] > 0):.2f}"),
                                          (un_masks[i], f"U-Net, Dice {dice(un_masks[i], L_te[i] > 0):.2f}")]):
            axes[r, c].imshow(img, cmap="gray"); axes[r, c].axis("off")
            axes[r, c].set_title(title, fontsize=8)
    fig.suptitle("Three hardest test images", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "examples.png", dpi=150)


if __name__ == "__main__":
    main()
