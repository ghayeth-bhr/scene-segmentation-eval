"""Mask IoU + zone-decision agreement for each prediction file, with the bbox-only baseline as reference.

For every ground-truth object:
  - match it to a prediction of the same image (Hungarian on mask IoU; unmatched -> IoU 0)
  - compare the in-zone decision of each method with the decision from the GT mask
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from masks import ann_to_mask, box_to_mask, decode, fraction_in_zone, iou, zone_mask


def match(gt_masks: list, pred_masks: list) -> tuple[list[float], dict[int, int]]:
    """IoU of each GT object with its matched prediction (0 when unmatched), and the GT->pred assignment."""
    if not gt_masks or not pred_masks:
        return [0.0] * len(gt_masks), {}
    m = np.array([[iou(g, p) for p in pred_masks] for g in gt_masks])
    rows, cols = linear_sum_assignment(-m)
    ious = [0.0] * len(gt_masks)
    assign = {}
    for r, c in zip(rows, cols):
        if m[r, c] > 0:
            ious[r] = float(m[r, c])
            assign[int(r)] = int(c)
    return ious, assign


def evaluate(coco: dict, zone: list, preds_by_method: dict, thr: float) -> dict:
    images = {im["id"]: im for im in coco["images"]}
    gt = defaultdict(list)
    for a in coco["annotations"]:
        gt[a["image_id"]].append(a)

    stats = {name: {"ious": [], "correct": 0, "flips_vs_bbox": 0} for name in ["bbox_only", *preds_by_method]}
    n_obj = 0
    for iid, anns in gt.items():
        im = images[iid]
        h, w = im["height"], im["width"]
        z = zone_mask(zone, h, w)
        gt_masks = [ann_to_mask(a, h, w) for a in anns]
        ref = [fraction_in_zone(m, z) > thr for m in gt_masks]
        n_obj += len(anns)

        # bbox-only baseline: the GT box itself, used as a mask (what box-based zone logic sees)
        box_masks = [box_to_mask(a["bbox"], h, w) for a in anns]
        box_dec = [fraction_in_zone(m, z) > thr for m in box_masks]
        stats["bbox_only"]["ious"] += [iou(g, b) for g, b in zip(gt_masks, box_masks)]
        stats["bbox_only"]["correct"] += sum(d == r for d, r in zip(box_dec, ref))

        for name, preds in preds_by_method.items():
            pm = [decode(p["segmentation"]) for p in preds.get(im["file_name"], [])]
            ious, assign = match(gt_masks, pm)
            stats[name]["ious"] += ious
            for k, r in enumerate(ref):
                dec = fraction_in_zone(pm[assign[k]], z) > thr if k in assign else False
                stats[name]["correct"] += dec == r
                stats[name]["flips_vs_bbox"] += dec != box_dec[k]

    table = {}
    for name, s in stats.items():
        table[name] = {
            "objects": n_obj,
            "mean_iou": round(float(np.mean(s["ious"])) if s["ious"] else 0.0, 4),
            "zone_accuracy": round(s["correct"] / n_obj, 4) if n_obj else 0.0,
            "decisions_changed_vs_bbox": None if name == "bbox_only" else s["flips_vs_bbox"],
        }
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--pred", nargs="*", default=[])
    ap.add_argument("--zone-thr", type=float, default=0.5)
    a = ap.parse_args()

    with open(a.ann, encoding="utf-8") as fh:
        coco = json.load(fh)
    with open(a.zone, encoding="utf-8") as fh:
        zone = json.load(fh)["zone"]
    preds_by_method = {}
    for p in a.pred:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        by_img = defaultdict(list)
        for x in d["predictions"]:
            by_img[x["file_name"]].append(x)
        preds_by_method[d["method"]] = by_img

    table = evaluate(coco, zone, preds_by_method, a.zone_thr)
    lines = ["| Method | Objects | Mean mask IoU | Zone decision accuracy | Decisions changed vs. bbox |",
             "|---|---|---|---|---|"]
    for name, r in table.items():
        changed = "—" if r["decisions_changed_vs_bbox"] is None else r["decisions_changed_vs_bbox"]
        lines.append(f'| {name} | {r["objects"]} | {r["mean_iou"]} | {r["zone_accuracy"]} | {changed} |')
    os.makedirs("results", exist_ok=True)
    with open("results/summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open("results/summary.json", "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
