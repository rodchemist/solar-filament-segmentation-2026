# Language: Python 3.12
# Lines of Code: 111
# File: src/baseline.py
# Version: 1.0.0
# Project: solar-filament-segmentation-2026
# Repository: solar-filament-segmentation-2026
# Author: Rod Sanchez
# Created: 2026-10-20 00:36
# Last Edited: 2026-10-20 20:30
"""Filament segmentation 2026 (MAGFiLO, GONG H-alpha): classical, training-free baseline.

Filaments are dark, elongated features on the solar disk. Per image, at 1/4 resolution:
  1. solar disk = largest bright component, eroded so the dark limb is excluded;
  2. flatten limb darkening: ratio = image / Gaussian-blurred image (inside the disk);
  3. candidate pixels: ratio < T; small closing; connected components with area >= A (at 1/4 scale) are filaments;
  4. upscale each component to 2048 x 2048 and encode as COCO compressed RLE counts (the leaderboard format).
T and A are chosen by Panoptic Quality (IoU > 0.5 matching) on a fixed subset of training images.
usage: baseline.py tune [--n 60] | predict --T 0.88 --A 30"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from pycocotools import mask as cmask
from skimage.draw import polygon as draw_polygon

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/MAGFiLO_1.0_Kaggle_2026"
S = 4                      # downscale factor
N = 2048 // S


def load(path):
    im = Image.open(path).convert("L").resize((N, N), Image.BILINEAR)
    return np.asarray(im, np.float32)


def segment(img, T, A, sigma=20.0):
    disk = img > 0.25 * img.max()
    lab, n = ndi.label(disk)
    if n == 0:
        return np.zeros_like(img, np.int32), 0
    disk = lab == (np.argmax(ndi.sum(disk, lab, range(1, n + 1))) + 1)
    disk = ndi.binary_fill_holes(disk)
    core = ndi.binary_erosion(disk, iterations=6)
    w = ndi.gaussian_filter(disk.astype(np.float32), sigma)
    bg = ndi.gaussian_filter(np.where(disk, img, 0), sigma) / np.maximum(w, 1e-3)
    ratio = np.where(core, img / np.maximum(bg, 1e-3), 1.0)
    cand = ndi.binary_closing(ratio < T, iterations=1) & core
    lab, n = ndi.label(cand)
    if n == 0:
        return lab, 0
    areas = ndi.sum(cand, lab, range(1, n + 1))
    keep = np.flatnonzero(areas >= A) + 1
    out = np.zeros_like(lab)
    for k, l in enumerate(keep, 1):
        out[lab == l] = k
    return out, len(keep)


def pq(gt_masks, pred_lab, n_pred):
    """Panoptic Quality for one image: unique matching at IoU > 0.5 (PQ property)."""
    preds = [pred_lab == k for k in range(1, n_pred + 1)]
    used, tp_iou = set(), []
    for g in gt_masks:
        gs = g.sum()
        for j, p in enumerate(preds):
            if j in used:
                continue
            inter = np.logical_and(g, p).sum()
            if inter == 0:
                continue
            iou = inter / (gs + p.sum() - inter)
            if iou > 0.5:
                used.add(j); tp_iou.append(iou); break
    tp = len(tp_iou); fp = n_pred - tp; fn = len(gt_masks) - tp
    return sum(tp_iou), tp, fp, fn


def gt_by_image():
    d = json.load(open(DATA / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    files = {im["id"]: im["file_name"] for im in d["images"]}
    by = {}
    for a in d["annotations"]:
        seg = json.loads(a["segmentation"]) if isinstance(a["segmentation"], str) else a["segmentation"]
        by.setdefault(a["image_id"], []).append(seg)
    have = {p.name for p in (DATA / "train/train_images").iterdir()}
    return {files[i]: s for i, s in by.items() if files.get(i) in have}


def rasterize(segs):
    out = []
    for seg in segs:
        polys = seg if isinstance(seg[0], list) else [seg]
        m = np.zeros((N, N), bool)
        for p in polys:
            xy = np.asarray(p, np.float64).reshape(-1, 2) / S
            rr, cc = draw_polygon(xy[:, 1], xy[:, 0], (N, N)); m[rr, cc] = True
        if m.any():
            out.append(m)
    return out


def tune(n):
    gt = gt_by_image(); names = sorted(gt)[:: max(1, len(gt) // n)][:n]
    cache = {f: (load(DATA / "train/train_images" / f), rasterize(gt[f])) for f in names}
    best = None
    for T in (0.80, 0.84, 0.88, 0.91, 0.94):
        for A in (10, 30, 60, 120):
            s = np.zeros(4)
            for f, (img, g) in cache.items():
                lab, k = segment(img, T, A); s += pq(g, lab, k)
            score = s[0] / max(1e-9, s[1] + 0.5 * s[2] + 0.5 * s[3])
            print(f"T {T:.2f} A {A:4d}  PQ {score:.4f}  TP {int(s[1])} FP {int(s[2])} FN {int(s[3])}", flush=True)
            if best is None or score > best[0]:
                best = (score, T, A)
    print("best", best)
    json.dump(dict(pq=best[0], T=best[1], A=best[2], n_images=len(cache)), open(ROOT / "results_tune.json", "w"), indent=1)


def predict(T, A):
    rows = []
    for p in sorted((DATA / "test/test_images").iterdir()):
        img = load(p); lab, k = segment(img, T, A); stem = p.stem
        big = np.kron(lab, np.ones((S, S), np.int32))
        for j in range(1, k + 1):
            rle = cmask.encode(np.asfortranarray((big == j).astype(np.uint8)))
            rows.append((f"{stem}_{j}", rle["counts"].decode("ascii")))
    out = ROOT / "submissions" / f"submission_thresh_T{T}_A{A}.csv"; out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(out, index=False)
    print("wrote", out, len(rows), "filaments over", len({r[0].rsplit('_', 1)[0] for r in rows}), "images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--T", type=float, default=0.88); ap.add_argument("--A", type=int, default=30)
    a = ap.parse_args()
    tune(a.n) if a.mode == "tune" else predict(a.T, a.A)
