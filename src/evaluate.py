#!/usr/bin/env python3
"""Filament segmentation 2026: evaluation beyond the leaderboard number, on the fixed val split.
Reports, for a post-processing configuration applied to cached val probabilities (see postproc.py probs):
  PQ = SQ x RQ (merged annotation sets, as the public LB; and per annotation set), the IoU and Dice distributions of the
  matched pairs, the one-to-many relations (one prediction spanning several GT filaments, one GT split over several
  predictions), the never-touched FP/FN, and the inter-annotator PQ on the files that carry two annotation sets.
usage: evaluate.py --tags v3_full_crop1024,v3_full_crop1024_s2 --thr 0.5 --min-area 240 --merge-r 12 [--tta] [--out results_eval.json]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unet, postproc


def relations(gt_masks, lab, n, min_frac=0.1):
    """Match stats for one image against one GT list. Returns dict of counts and per-pair IoU/Dice of matches."""
    out = dict(tp_iou=[], tp_dice=[], fp_none=0, fp_split=0, fp_merge=0, fp_shape=0, fn_none=0, fn_split=0, fn_other=0)
    if n == 0:
        out["fn_none"] = len(gt_masks); return out
    if not gt_masks:
        out["fp_none"] = n; return out
    pa = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    G = np.stack(gt_masks).reshape(len(gt_masks), -1); ga = G.sum(1)
    inter = np.stack([np.bincount(lab.ravel()[G[k]], minlength=n + 1)[1:] for k in range(len(gt_masks))])
    iou = inter / (ga[:, None] + pa[None, :] - inter); dice = 2 * inter / (ga[:, None] + pa[None, :])
    matched_p, matched_g = set(), set()
    for k in range(len(gt_masks)):                       # greedy in GT order, same as the scorer
        for j in np.flatnonzero(iou[k] > 0.5):
            if j not in matched_p:
                matched_p.add(j); matched_g.add(k); out["tp_iou"].append(float(iou[k, j])); out["tp_dice"].append(float(dice[k, j])); break
    for j in range(n):
        if j in matched_p:
            continue
        gs = np.flatnonzero(inter[:, j] / pa[j] >= min_frac)
        if len(gs) == 0:
            out["fp_none"] += 1
        elif len(gs) >= 2:
            out["fp_split"] += 1                          # one prediction spans >= 2 GT filaments
        elif (inter[gs[0]] / ga[gs[0]] >= min_frac).sum() >= 2:
            out["fp_merge"] += 1                          # this prediction is one of several pieces of one GT
        else:
            out["fp_shape"] += 1
    for k in range(len(gt_masks)):
        if k in matched_g:
            continue
        js = np.flatnonzero(inter[k] / ga[k] >= min_frac)
        if (inter[k] > 0).sum() == 0:
            out["fn_none"] += 1
        elif len(js) >= 2:
            out["fn_split"] += 1                          # GT covered by >= 2 predictions
        else:
            out["fn_other"] += 1
    return out


def summarize(rows):
    s = {k: sum(r[k] for r in rows) for k in rows[0] if not isinstance(rows[0][k], list)}
    ious = [x for r in rows for x in r["tp_iou"]]; dices = [x for r in rows for x in r["tp_dice"]]
    tp = len(ious); fp = s["fp_none"] + s["fp_split"] + s["fp_merge"] + s["fp_shape"]; fn = s["fn_none"] + s["fn_split"] + s["fn_other"]
    sq = float(np.mean(ious)) if tp else 0.0; rq = tp / max(1e-9, tp + 0.5 * fp + 0.5 * fn)
    return dict(PQ=sq * rq, SQ=sq, RQ=rq, TP=tp, FP=fp, FN=fn, **s,
                iou_percentiles={p: float(np.percentile(ious, p)) for p in (10, 25, 50, 75, 90)} if tp else {},
                dice_percentiles={p: float(np.percentile(dices, p)) for p in (10, 25, 50, 75, 90)} if tp else {},
                iou_hist=np.histogram(ious, bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])[0].tolist() if tp else [])


def main(a):
    names, imgs, _ = unet.load_cache(); _, val = unet.split(names)
    idx = {f: i for i, f in enumerate(names)}; va = np.array([idx[f] for f in val])
    probs = postproc.load_probs(a.tags, a.tta); by = unet.annotations()
    merged, per_set, human = [], [], []
    for i, f in enumerate(val):
        lab, n = postproc.instances_v2(probs[i], imgs[va[i]], a.thr, a.min_area, a.limb, a.merge_r, a.min_mean)
        sets = [unet.rasterize(by[f][sid]) for sid in sorted(by[f])]
        merged.append(relations([m for s in sets for m in s], lab, n))
        per_set.extend(relations(s, lab, n) for s in sets)
        if len(sets) >= 2:                                 # annotator 2 as "prediction" against annotator 1
            lab2 = np.zeros_like(lab)
            for k, m in enumerate(sets[1], 1):
                lab2[m] = k
            human.append(relations(sets[0], lab2, len(sets[1])))
    res = dict(tags=a.tags, tta=a.tta, config=dict(thr=a.thr, min_area=a.min_area, limb=a.limb, merge_r=a.merge_r, min_mean=a.min_mean),
               n_val_files=len(val), merged=summarize(merged), per_set=summarize(per_set),
               inter_annotator=dict(n_files=len(human), **summarize(human)) if human else None)
    for k in ("merged", "per_set", "inter_annotator"):
        r = res[k]
        if r:
            print(f"{k:16s} PQ {r['PQ']:.4f} = SQ {r['SQ']:.3f} x RQ {r['RQ']:.3f} | TP {r['TP']} FP {r['FP']} FN {r['FN']} | "
                  f"FP: none {r['fp_none']} spans>=2GT {r['fp_split']} piece-of-GT {r['fp_merge']} shape {r['fp_shape']} | "
                  f"FN: none {r['fn_none']} split>=2pred {r['fn_split']} other {r['fn_other']} | IoU p50 {r['iou_percentiles'].get(50, 0):.3f} Dice p50 {r['dice_percentiles'].get(50, 0):.3f}")
    json.dump(res, open(unet.ROOT / a.out, "w"), indent=1); print("wrote", unet.ROOT / a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", default="v3_full_crop1024,v3_full_crop1024_s2"); ap.add_argument("--tta", action="store_true")
    ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--min-area", type=int, default=240); ap.add_argument("--limb", type=int, default=0)
    ap.add_argument("--merge-r", type=int, default=12); ap.add_argument("--min-mean", type=float, default=0.0); ap.add_argument("--out", default="results_eval.json")
    a = ap.parse_args(); a.tags = a.tags.split(","); main(a)
