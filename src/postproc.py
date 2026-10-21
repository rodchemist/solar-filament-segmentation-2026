#!/usr/bin/env python3
"""Filament segmentation 2026: post-processing of U-Net probability maps, tuned on the val split.
Hardware Target: RTX 4090 (shared), Ryzen 9 7950X, 128GB DDR5

Instance rule (instances_v2), all at 1/2 scale:
  1. binary = prob > thr, restricted to the solar disk eroded by `limb` px (drops limb artefacts);
  2. fragments closer than 2*merge_r px are merged (dilate by merge_r, label, map the original pixels);
  3. components with area < min_area or mean probability < min_mean are dropped.
Selection metric: merged-GT PQ (the public LB tracks it); per-set PQ reported as secondary.

usage: postproc.py probs --tags v1[,v2]        # cache val probabilities (GPU) for the tags, with and without TTA
       postproc.py sweep --tags v1 [--tta]     # staged one-factor-at-a-time sweep on val -> results_post_<tags>.json
       postproc.py predict --tags v1 --thr .. --min-area .. --limb .. --merge-r .. --min-mean .. [--tta]
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unet
from unet import ROOT, DATA, CACHE, S, N


def disk_mask(img_uint8, erode):
    img = img_uint8.astype(np.float32)
    disk = img > 0.25 * img.max()
    lab, n = ndi.label(disk)
    if n == 0:
        return np.ones_like(disk)
    disk = lab == (np.argmax(ndi.sum(disk, lab, range(1, n + 1))) + 1)
    disk = ndi.binary_fill_holes(disk)
    return ndi.binary_erosion(disk, iterations=erode) if erode > 0 else disk


def instances_v2(prob, img, thr, min_area, limb=0, merge_r=0, min_mean=0.0):
    binary = prob > thr
    if limb > 0:
        binary &= disk_mask(img, limb)
    if merge_r > 0:
        lab, n = ndi.label(ndi.binary_dilation(binary, iterations=merge_r))
        lab = np.where(binary, lab, 0)
    else:
        lab, n = ndi.label(binary)
    if n == 0:
        return lab.astype(np.int32), 0
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    keep = areas[1:] >= min_area
    if min_mean > 0:
        sums = np.bincount(lab.ravel(), weights=prob.ravel().astype(np.float64), minlength=n + 1)
        keep &= (sums[1:] / np.maximum(areas[1:], 1)) >= min_mean
    keep = np.flatnonzero(keep) + 1
    remap = np.zeros(n + 1, np.int32); remap[keep] = np.arange(1, len(keep) + 1)
    return remap[lab], len(keep)


# ----------------------------------------------------------------------------- val probs cache
def cache_probs(tags):
    device = unet.setup_cuda(); names, imgs, _ = unet.load_cache(); _, val = unet.split(names)
    idx = {f: i for i, f in enumerate(names)}; va = np.array([idx[f] for f in val])
    for tag in tags:
        model, ck = unet.load_model(tag, device)
        for tta in (False, True):
            p = unet.predict_probs(model, imgs[va], device, tta=tta)
            np.save(CACHE / f"val_probs_{tag}{'_tta' if tta else ''}.npy", p)
            print("cached", tag, "tta", tta, p.shape, "epoch", ck["epoch"], flush=True)


def load_probs(tags, tta):
    """Mean of the cached val probability maps of the given tags (model averaging)."""
    ps = []
    for t in tags:
        f = CACHE / f"val_probs_{t}{'_tta' if tta else ''}.npy"
        if f.exists():
            ps.append(np.load(f).astype(np.float32)); continue
        # cross-scale: a 1/2-scale model's probabilities used at full resolution (nearest upsampling)
        g = DATA.parent / "_cache" / f.name
        assert S == 1 and g.exists(), f"missing {f}"
        ps.append(np.repeat(np.repeat(np.load(g), 2, axis=1), 2, axis=2).astype(np.float32)); print("upsampled", t, flush=True)
    return np.mean(ps, 0) if len(ps) > 1 else ps[0]


# ----------------------------------------------------------------------------- sweep
_G = {}


def _score_one(args):
    i, cfg = args
    lab, n = instances_v2(_G["probs"][i], _G["imgs"][i], cfg["thr"], cfg["min_area"], cfg["limb"], cfg["merge_r"], cfg["min_mean"])
    f = _G["val"][i]; g = _G["gts"][f]
    return unet.pq_image(g[0], lab, n), [unet.pq_image(s, lab, n) for s in g[1]]


def score(cfg, pool):
    out = pool.map(_score_one, [(i, cfg) for i in range(len(_G["val"]))])
    merged = unet.pq_total([m for m, _ in out]); per_set = unet.pq_total([x for _, ss in out for x in ss])
    return dict(cfg, pq_merged=float(merged[0]), tp=merged[1], fp=merged[2], fn=merged[3],
                pq_set=float(per_set[0]), tp_set=per_set[1], fp_set=per_set[2], fn_set=per_set[3])


def sweep(tags, tta, start):
    names, imgs, _ = unet.load_cache(); _, val = unet.split(names)
    idx = {f: i for i, f in enumerate(names)}; va = np.array([idx[f] for f in val])
    _G.update(probs=load_probs(tags, tta), imgs=imgs[va], val=val, gts=unet.val_gt(val))
    key = "+".join(tags) + ("_tta" if tta else "") + ("" if S == 2 else f"_s{S}")
    rows = []
    with Pool(26) as pool:
        best = score(dict(start), pool); rows.append(dict(best, stage="start")); show(best, "start")
        k1, k2 = 2 // S, 4 // (S * S)                     # grids are defined at 1/2 scale; rescale for full res
        stages = [("limb", [v * k1 for v in (0, 2, 4, 8, 12, 16)]), ("min_mean", [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]),
                  ("merge_r", [v * k1 for v in (0, 1, 2, 3, 4, 6, 8)]), ("min_area", [v * k2 for v in (20, 40, 60, 80, 100, 120, 160)]),
                  ("thr", [0.4, 0.5, 0.6, 0.7, 0.8])]
        for k, grid in stages:
            for v in grid:
                if v == best[k]:
                    continue
                cfg = {kk: best[kk] for kk in ("thr", "min_area", "limb", "merge_r", "min_mean")}; cfg[k] = v
                r = score(cfg, pool); rows.append(dict(r, stage=k)); show(r, k)
                if r["pq_merged"] > best["pq_merged"] + 1e-6:
                    best = r
            print(f"-- after {k}: best merged {best['pq_merged']:.4f} {cfgstr(best)}", flush=True)
    res = dict(tags=tags, tta=tta, best=best, rows=rows)
    json.dump(res, open(ROOT / f"results_post_{key}.json", "w"), indent=1)
    print("BEST", cfgstr(best), f"merged {best['pq_merged']:.4f} TP {best['tp']} FP {best['fp']} FN {best['fn']} | per-set {best['pq_set']:.4f}")


def cfgstr(c):
    return f"thr {c['thr']} min_area {c['min_area']} limb {c['limb']} merge_r {c['merge_r']} min_mean {c['min_mean']}"


def show(r, stage):
    print(f"[{stage:8s}] {cfgstr(r):60s} merged {r['pq_merged']:.4f} TP {r['tp']} FP {r['fp']} FN {r['fn']} | per-set {r['pq_set']:.4f} TP {r['tp_set']} FP {r['fp_set']} FN {r['fn_set']}", flush=True)


# ----------------------------------------------------------------------------- predict
def predict(a):
    import torch
    from pycocotools import mask as cmask
    from PIL import Image
    device = unet.setup_cuda()
    paths = sorted((DATA / "test/test_images").iterdir())
    imgs = np.stack([unet.load_image(p) for p in paths])          # at the output scale S
    probs = []
    for tag in a.tags:                                             # models may have been trained at another scale
        ck = torch.load(unet.CKPT / f"unet_{tag}_best.pt", map_location="cpu", weights_only=False)
        sc = ck.get("scale", 2); assert sc >= S, f"{tag}: trained at scale {sc}, output scale {S} is coarser"
        model = unet.build_model(pretrained=False, encoder=ck.get("args", {}).get("encoder", "resnet34")); model.load_state_dict(ck["model"]); model = model.to(device).to(memory_format=torch.channels_last)
        if sc == S:
            x = imgs
        else:
            n = 2048 // sc; x = np.stack([np.asarray(Image.open(p).convert("L").resize((n, n), Image.BILINEAR), np.uint8) for p in paths])
        pr = unet.predict_probs(model, x, device, bs=max(1, 4 * sc * sc // 4), tta=a.tta).astype(np.float32)
        if sc != S:
            pr = np.repeat(np.repeat(pr, sc // S, axis=1), sc // S, axis=2)
        probs.append(pr); print("predicted", tag, "scale", sc, "epoch", ck["epoch"], flush=True); del model
    probs = np.mean(probs, 0)
    rows = []
    for p, pr, im in zip(paths, probs, imgs):
        lab, k = instances_v2(pr, im, a.thr, a.min_area, a.limb, a.merge_r, a.min_mean)
        big = np.kron(lab, np.ones((S, S), np.int32))
        for j in range(1, k + 1):
            rle = cmask.encode(np.asfortranarray((big == j).astype(np.uint8)))
            rows.append((f"{p.stem}_{j}", rle["counts"].decode("ascii")))
    name = f"submission_post_{'+'.join(a.tags)}_s{S}_thr{a.thr}_a{a.min_area}_l{a.limb}_m{a.merge_r}_p{a.min_mean}{'_tta' if a.tta else ''}.csv"
    out = ROOT / "submissions" / name
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(out, index=False)
    print("wrote", out, len(rows), "filaments over", len({r[0].rsplit('_', 1)[0] for r in rows}), "images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["probs", "sweep", "predict"])
    ap.add_argument("--tags", default="v1"); ap.add_argument("--tta", action="store_true")
    ap.add_argument("--thr", type=float, default=0.7); ap.add_argument("--min-area", type=int, default=80)
    ap.add_argument("--limb", type=int, default=0); ap.add_argument("--merge-r", type=int, default=0); ap.add_argument("--min-mean", type=float, default=0.0)
    a = ap.parse_args(); a.tags = a.tags.split(",")
    if a.mode == "probs":
        cache_probs(a.tags)
    elif a.mode == "sweep":
        sweep(a.tags, a.tta, dict(thr=a.thr, min_area=a.min_area, limb=a.limb, merge_r=a.merge_r, min_mean=a.min_mean))
    else:
        predict(a)
