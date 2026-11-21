"""Lightweight instance segmentation with Ultralytics YOLO-seg -> predictions/yolo_seg.json (COCO RLE)."""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from masks import encode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="keep only these class ids")
    ap.add_argument("--out", default="predictions/yolo_seg.json")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    preds, times = [], []
    files = sorted(glob.glob(os.path.join(a.images, "*.jpg")) + glob.glob(os.path.join(a.images, "*.png")))
    for f in files:
        h, w = cv2.imread(f).shape[:2]
        t0 = time.perf_counter()
        r = model.predict(f, conf=a.conf, classes=a.classes, retina_masks=True, verbose=False)[0]
        times.append((time.perf_counter() - t0) * 1000)
        if r.masks is None:
            continue
        masks = r.masks.data.cpu().numpy() > 0.5          # [N, H, W] at original resolution
        for m, box, score, cls in zip(masks, r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                      r.boxes.cls.cpu().numpy()):
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            x1, y1, x2, y2 = box.tolist()
            preds.append({"file_name": os.path.basename(f), "score": float(score), "class_id": int(cls),
                          "bbox": [x1, y1, x2 - x1, y2 - y1], "segmentation": encode(m)})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"method": "yolo_seg", "model": a.model,
                   "ms_per_image": round(float(np.median(times[1:] or times)), 1) if times else None,
                   "predictions": preds}, fh)
    print(f"{len(preds)} masks on {len(files)} images -> {a.out}")


if __name__ == "__main__":
    main()
