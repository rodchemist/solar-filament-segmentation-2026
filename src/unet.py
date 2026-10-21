#!/usr/bin/env python3
"""Filament segmentation 2026 (MAGFiLO, GONG H-alpha): trained U-Net baseline.
Hardware Target: RTX 4090 (shared, <16 GB), Ryzen 9 7950X, 128GB DDR5

Semantic U-Net (ResNet-34 encoder, ImageNet init, 1-channel input) trained at 1/2 resolution
(1024 x 1024) on the 707 annotated training images (all annotation sets of an image are merged
into one binary mask). Instances = connected components of the thresholded probability map at
1/2 scale, filtered by a minimum area, upscaled x2 and encoded as COCO compressed RLE.

Validation: fixed split by capture date (stem[:8]) of the 707 files, seed 20261020, ~15 % of dates in val, scored
with the same Panoptic Quality as src/baseline.py (greedy unique matching, IoU > 0.5) against
the merged annotation sets of each file (baseline-compatible, duplicates count as FN), and also
per annotation set (each (file, set) pair scored as its own image, TP/FP/FN aggregated).

usage:
  unet.py cache                       # rasterise masks + resize images once -> data/_cache
  unet.py train --tag v1 --epochs 40  # train, checkpoints/unet_<tag>_{best,last}.pt
  unet.py eval  --tag v1              # val sweep of threshold x min-area, writes results_unet_<tag>.json
  unet.py predict --tag v1 --thr 0.5 --min-area 40 [--tta]   # submissions/submission_unet_<tag>.csv
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_polygon

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FILAMENT_DATA", ROOT / "data/MAGFiLO_1.0_Kaggle_2026"))   # train/ and test/ live here
S = int(os.environ.get("FILAMENT_SCALE", "2"))   # downscale factor: 2 = 1024x1024 (default), 1 = full 2048x2048
N = 2048 // S
CACHE = Path(os.environ.get("FILAMENT_CACHE", ROOT / "data")) / ("_cache" if S == 2 else f"_cache_s{S}")   # gitignored
CKPT = ROOT / "checkpoints"
SEED = 20261020
VAL_FRAC = 0.15
GPU_MEM_FRACTION = 16 / 24                 # shared 4090: hard cap at ~16 GB


# ----------------------------------------------------------------------------- data
def annotations():
    d = json.load(open(DATA / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    files = {im["id"]: im["file_name"] for im in d["images"]}
    by_set = {}                                     # image id (annotation set) -> list of polygons
    for a in d["annotations"]:
        seg = json.loads(a["segmentation"]) if isinstance(a["segmentation"], str) else a["segmentation"]
        by_set.setdefault(a["image_id"], []).append(seg)
    by_file = {}                                    # file name -> {set id: polygons}
    for sid, segs in by_set.items():
        by_file.setdefault(files[sid], {})[sid] = segs
    return by_file


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


def load_image(path):
    return np.asarray(Image.open(path).convert("L").resize((N, N), Image.BILINEAR), np.uint8)


def split(names):
    """By capture date (first 8 chars, YYYYMMDD): several observatories image the same Sun on the
    same day, so a by-file split would put near-duplicates on both sides."""
    names = sorted(names); dates = sorted({f[:8] for f in names})
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(dates))
    val_dates = {dates[i] for i in perm[:int(round(VAL_FRAC * len(dates)))]}
    val = [f for f in names if f[:8] in val_dates]; train = [f for f in names if f[:8] not in val_dates]
    return train, val


def build_cache():
    CACHE.mkdir(exist_ok=True)
    by_file = annotations(); names = sorted(by_file)
    imgs = np.zeros((len(names), N, N), np.uint8); masks = np.zeros((len(names), N, N), np.uint8)
    t = time.time()
    for i, f in enumerate(names):
        imgs[i] = load_image(DATA / "train/train_images" / f)
        for segs in by_file[f].values():
            for m in rasterize(segs):
                masks[i] |= m
        if i % 100 == 0:
            print(f"cache {i}/{len(names)} {time.time() - t:.0f}s", flush=True)
    np.save(CACHE / "train_imgs.npy", imgs); np.save(CACHE / "train_masks.npy", masks)
    json.dump(names, open(CACHE / "names.json", "w"))
    tr, va = split(names)
    json.dump({"seed": SEED, "val_frac": VAL_FRAC, "unit": "capture date (file stem[:8])", "n_dates": len({f[:8] for f in names}),
               "val_dates": sorted({f[:8] for f in va}), "train": tr, "val": va}, open(ROOT / "src/split_seed20261020.json", "w"), indent=0)
    print("cached", len(names), "images; train", len(tr), "val", len(va), "mask fraction %.4f" % masks.mean())


def load_cache():
    names = json.load(open(CACHE / "names.json"))
    return names, np.load(CACHE / "train_imgs.npy"), np.load(CACHE / "train_masks.npy")


# ----------------------------------------------------------------------------- metric
def pq_image(gt_masks, pred_lab, n_pred):
    """Same semantics as baseline.pq: greedy over GT in order, first unused pred (label order)
    with IoU > 0.5 is the match. Intersections restricted to the GT bbox for speed."""
    if n_pred == 0:
        return 0.0, 0, 0, len(gt_masks)
    pred_area = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)
    used, tp_iou = set(), []
    for g in gt_masks:
        gs = int(g.sum())
        ys, xs = np.nonzero(g.any(1))[0], np.nonzero(g.any(0))[0]
        sub = pred_lab[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1][g[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]]
        inter = np.bincount(sub, minlength=n_pred + 1)
        for j in np.flatnonzero(inter[1:] > 0) + 1:
            if j in used:
                continue
            iou = inter[j] / (gs + pred_area[j] - inter[j])
            if iou > 0.5:
                used.add(j); tp_iou.append(iou); break
    tp = len(tp_iou); fp = n_pred - tp; fn = len(gt_masks) - tp
    return float(sum(tp_iou)), tp, fp, fn


def pq_total(stats):
    s = np.asarray(stats, float).sum(0)
    return s[0] / max(1e-9, s[1] + 0.5 * s[2] + 0.5 * s[3]), int(s[1]), int(s[2]), int(s[3])


def instances(prob, thr, min_area):
    lab, n = ndi.label(prob > thr)
    if n == 0:
        return lab, 0
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    keep = np.flatnonzero(areas >= min_area) + 1
    remap = np.zeros(n + 1, np.int32); remap[keep] = np.arange(1, len(keep) + 1)
    return remap[lab], len(keep)


def val_gt(val_names, by_file=None):
    """Per val file: (merged masks over all annotation sets, [masks of each set])."""
    by_file = by_file or annotations()
    out = {}
    for f in val_names:
        sets = [rasterize(by_file[f][sid]) for sid in sorted(by_file[f])]
        out[f] = ([m for s in sets for m in s], sets)
    return out


# ----------------------------------------------------------------------------- model
def build_model(pretrained=True, encoder="resnet34"):
    import torch, torch.nn as nn, torchvision

    class Up(nn.Module):
        def __init__(self, cin, cskip, cout):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        def forward(self, x, skip):
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat([x, skip], 1)
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            r = getattr(torchvision.models, encoder)(weights="IMAGENET1K_V1" if pretrained else None)
            c1, c2, c3, c4 = (64, 128, 256, 512) if encoder in ("resnet18", "resnet34") else (256, 512, 1024, 2048)
            w = r.conv1.weight.data.sum(1, keepdim=True)
            r.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False); r.conv1.weight.data = w
            self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)      # /2, 64
            self.pool = r.maxpool
            self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4   # /4 c1, /8 c2, /16 c3, /32 c4
            self.u4 = Up(c4, c3, 256); self.u3 = Up(256, c2, 128); self.u2 = Up(128, c1, 64)
            self.u1 = Up(64, 64, 32); self.u0 = Up(32, 0, 16)
            self.head = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            s0 = self.stem(x); s1 = self.l1(self.pool(s0)); s2 = self.l2(s1); s3 = self.l3(s2); s4 = self.l4(s3)
            x = self.u4(s4, s3); x = self.u3(x, s2); x = self.u2(x, s1); x = self.u1(x, s0); x = self.u0(x, None)
            return self.head(x)

    return UNet()


def normalize(x_uint8):
    return (x_uint8.float() / 255.0 - 0.5) / 0.25


def setup_cuda(cpu=False):
    import torch
    if cpu:
        return torch.device("cpu")
    assert torch.cuda.is_available(), "GPU required"
    torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION, 0)
    torch.backends.cudnn.benchmark = True
    try:
        sys.path.insert(0, str(ROOT.parents[1]))
        from scripts.gpu_awareness import check_gpu_usage
        check_gpu_usage(__file__)
    except Exception as e:                                 # advisory only
        print("gpu_awareness skipped:", e)
    return torch.device("cuda")


def augment(x, y, rng):
    import torch
    if rng.random() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if rng.random() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    k = int(rng.integers(4))
    if k:
        x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
    gain = float(rng.uniform(0.85, 1.15)); bias = float(rng.uniform(-0.15, 0.15))
    return x * gain + bias, y


def predict_probs(model, imgs_uint8, device, bs=4, tta=False):
    """imgs_uint8: (n, N, N) numpy -> probabilities (n, N, N) float16 numpy."""
    import torch
    out = np.zeros(imgs_uint8.shape, np.float16)
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for i in range(0, len(imgs_uint8), bs):
            x = normalize(torch.from_numpy(imgs_uint8[i:i + bs]).to(device)[:, None]).contiguous(memory_format=torch.channels_last)
            p = torch.sigmoid(model(x).float())
            if tta:
                p = p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1) + torch.sigmoid(model(x.flip(-2)).float()).flip(-2)
                p = p / 3
            out[i:i + bs] = p[:, 0].cpu().numpy().astype(np.float16)
    return out


def score_val(probs, val_names, gts, thr, min_area):
    """Returns (PQ vs merged sets, PQ per set): the per-set figure scores each (file, annotation
    set) pair as its own image and aggregates TP/FP/FN over all pairs (dataset-level PQ)."""
    merged, per_set = [], []
    for p, f in zip(probs, val_names):
        lab, n = instances(p.astype(np.float32), thr, min_area)
        merged.append(pq_image(gts[f][0], lab, n))
        per_set.extend(pq_image(g, lab, n) for g in gts[f][1])
    return pq_total(merged), pq_total(per_set)


# ----------------------------------------------------------------------------- train / eval / predict
def train(a):
    import torch
    device = setup_cuda(a.cpu); CKPT.mkdir(exist_ok=True); (ROOT / "logs").mkdir(exist_ok=True)
    log = open(ROOT / "logs" / f"unet_{a.tag}.log", "a")
    def P(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); log.write(msg + "\n"); log.flush()

    names, imgs, masks = load_cache(); tr_names, val_names = split(names)
    if a.limit:
        tr_names, val_names = tr_names[:a.limit], val_names[:max(1, a.limit // 2)]
    idx = {f: i for i, f in enumerate(names)}
    tr_idx = np.array([idx[f] for f in tr_names]); va_idx = np.array([idx[f] for f in val_names])
    gts = val_gt(val_names)
    P(f"== train {a.tag}: {a.encoder}, {len(tr_idx)} train / {len(va_idx)} val files, {N}x{N} crop {a.crop or 'none'}, bs {a.bs}, epochs {a.epochs}, lr {a.lr}, train seed {a.seed}, split seed {SEED}, min_area {a.min_area}")
    X = torch.from_numpy(imgs); Y = torch.from_numpy(masks)        # CPU uint8, pinned per batch

    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)       # training seed; the split seed stays SEED
    model = build_model(pretrained=not a.scratch, encoder=a.encoder).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(tr_idx) / a.bs); total = steps_per_epoch * a.epochs; warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    bce = torch.nn.BCEWithLogitsLoss()

    best, t0, step = -1.0, time.time(), 0
    for ep in range(a.epochs):
        model.train(); order = rng.permutation(tr_idx); tl = 0.0; te = time.time()
        for b in range(0, len(order), a.bs):
            ids = torch.from_numpy(order[b:b + a.bs])
            x = normalize(X[ids].to(device, non_blocking=True)[:, None]); y = (Y[ids].to(device, non_blocking=True)[:, None] > 0).float()
            if a.crop and a.crop < N:                       # random square crop per batch (full-res training)
                oy, ox = rng.integers(0, N - a.crop + 1, 2); x = x[..., oy:oy + a.crop, ox:ox + a.crop]; y = y[..., oy:oy + a.crop, ox:ox + a.crop]
            x, y = augment(x, y, rng); x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logit = model(x)
            logit = logit.float(); p = torch.sigmoid(logit)
            dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
            loss = bce(logit, y) + dice.mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            tl += loss.item() * len(ids)
        tl /= len(tr_idx)
        probs = predict_probs(model, imgs[va_idx], device, bs=max(1, a.bs * S * S // 4))
        (pq_m, tp, fp, fn), (pq_f, tpf, fpf, fnf) = score_val(probs, val_names, gts, 0.5, a.min_area)
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        P(f"ep {ep + 1:3d}/{a.epochs} loss {tl:.4f} valPQ(merged) {pq_m:.4f} TP {tp} FP {fp} FN {fn} | valPQ(per set) {pq_f:.4f} TP {tpf} FP {fpf} FN {fnf} | "
          f"{time.time() - te:.0f}s/ep total {(time.time() - t0) / 60:.1f}min peakmem {mem:.1f}GB")
        state = {"model": model.state_dict(), "epoch": ep + 1, "val_pq_merged": pq_m, "val_pq_set": pq_f, "tag": a.tag,
                 "args": vars(a), "scale": S, "split_seed": SEED, "time_min": (time.time() - t0) / 60}
        torch.save(state, CKPT / f"unet_{a.tag}_last.pt")
        if pq_m > best:
            best = pq_m; torch.save(state, CKPT / f"unet_{a.tag}_best.pt"); P(f"   saved best (epoch {ep + 1}, PQ {best:.4f})")
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            P("time cap reached, stopping"); break
    P(f"done: best valPQ(merged) {best:.4f}, {(time.time() - t0) / 60:.1f} min, checkpoint {CKPT / f'unet_{a.tag}_best.pt'}")


def load_model(tag, device):
    import torch
    ck = torch.load(CKPT / f"unet_{tag}_best.pt", map_location="cpu", weights_only=False)
    assert ck.get("scale", 2) == S, f"checkpoint {tag} was trained at scale {ck.get('scale')}, but FILAMENT_SCALE={S}"
    model = build_model(pretrained=False, encoder=ck.get("args", {}).get("encoder", "resnet34")); model.load_state_dict(ck["model"])
    return model.to(device).to(memory_format=torch.channels_last), ck


def evaluate(a):
    import torch
    device = setup_cuda(a.cpu); names, imgs, _ = load_cache(); _, val_names = split(names)
    if a.limit:
        val_names = val_names[:max(1, a.limit // 2)]
    idx = {f: i for i, f in enumerate(names)}; va_idx = np.array([idx[f] for f in val_names]); gts = val_gt(val_names)
    model, ck = load_model(a.tag, device)
    res = {"tag": a.tag, "epoch": ck["epoch"], "n_val": len(val_names), "sweep": []}
    for tta in ([False, True] if a.tta else [False]):
        probs = predict_probs(model, imgs[va_idx], device, bs=max(1, 4 * S * S // 4), tta=tta)
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            for ma in (m * 4 // (S * S) for m in (10, 20, 40, 60, 80, 120, 160)):
                (pq_m, tp, fp, fn), (pq_f, tpf, fpf, fnf) = score_val(probs, val_names, gts, thr, ma)
                res["sweep"].append(dict(tta=tta, thr=thr, min_area=ma, pq_merged=float(pq_m), tp=tp, fp=fp, fn=fn, pq_set=float(pq_f), tp_set=tpf, fp_set=fpf, fn_set=fnf))
                print(f"tta {int(tta)} thr {thr:.1f} min_area {ma:4d}  PQ(merged) {pq_m:.4f} TP {tp} FP {fp} FN {fn} | PQ(per-set) {pq_f:.4f} TP {tpf} FP {fpf} FN {fnf}", flush=True)
    b = max(res["sweep"], key=lambda r: r["pq_merged"]); res["best"] = b
    json.dump(res, open(ROOT / f"results_unet_{a.tag}.json", "w"), indent=1)
    print("best", b)


def predict(a):
    import torch
    from pycocotools import mask as cmask
    device = setup_cuda(a.cpu); model, ck = load_model(a.tag, device)
    paths = sorted((DATA / "test/test_images").iterdir())[:a.limit or None]
    imgs = np.stack([load_image(p) for p in paths])
    probs = predict_probs(model, imgs, device, bs=max(1, 4 * S * S // 4), tta=a.tta)
    rows = []
    for p, pr in zip(paths, probs):
        lab, k = instances(pr.astype(np.float32), a.thr, a.min_area)
        big = np.kron(lab, np.ones((S, S), np.int32))
        for j in range(1, k + 1):
            rle = cmask.encode(np.asfortranarray((big == j).astype(np.uint8)))
            rows.append((f"{p.stem}_{j}", rle["counts"].decode("ascii")))
    out = ROOT / "submissions" / f"submission_unet_{a.tag}_thr{a.thr}_a{a.min_area}{'_tta' if a.tta else ''}.csv"; out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(out, index=False)
    print("wrote", out, len(rows), "filaments over", len({r[0].rsplit('_', 1)[0] for r in rows}), "images (epoch", ck["epoch"], ")")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["cache", "train", "eval", "predict"])
    ap.add_argument("--tag", default="v1"); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--min-area", type=int, default=40); ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true"); ap.add_argument("--scratch", action="store_true"); ap.add_argument("--max-minutes", type=float, default=200)
    ap.add_argument("--seed", type=int, default=SEED, help="training seed (split seed is fixed)")
    ap.add_argument("--crop", type=int, default=0, help="random square crop size for training (0 = whole image)")
    ap.add_argument("--encoder", default="resnet34", help="torchvision resnet encoder: resnet18/34/50/101")
    ap.add_argument("--cpu", action="store_true", help="smoke test on CPU"); ap.add_argument("--limit", type=int, default=0, help="use only the first N files (smoke test)")
    a = ap.parse_args()
    {"cache": lambda a: build_cache(), "train": train, "eval": evaluate, "predict": predict}[a.mode](a)
