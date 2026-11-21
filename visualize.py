"""Overlay GT mask (green), predicted mask (red), GT box (yellow) and zone (blue) for every object."""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from masks import ann_to_mask, decode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="viz")
    a = ap.parse_args()

    coco = json.load(open(a.ann, encoding="utf-8"))
    zone = np.array(json.load(open(a.zone, encoding="utf-8"))["zone"], dtype=np.int32)
    preds = defaultdict(list)
    for p in json.load(open(a.pred, encoding="utf-8"))["predictions"]:
        preds[p["file_name"]].append(p)
    anns = defaultdict(list)
    for x in coco["annotations"]:
        anns[x["image_id"]].append(x)

    os.makedirs(a.out, exist_ok=True)
    for im in coco["images"]:
        img = cv2.imread(os.path.join(a.images, im["file_name"]))
        h, w = img.shape[:2]
        overlay = img.copy()
        for x in anns[im["id"]]:
            overlay[ann_to_mask(x, h, w)] = (0, 200, 0)
            bx, by, bw, bh = map(int, x["bbox"])
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 220, 220), 2)
        for p in preds[im["file_name"]]:
            overlay[decode(p["segmentation"])] = (0, 0, 220)
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
        cv2.polylines(img, [zone], True, (220, 80, 0), 2)
        cv2.imwrite(os.path.join(a.out, im["file_name"]), img)
    print(f"wrote {len(coco['images'])} overlays to {a.out}/")


if __name__ == "__main__":
    main()
