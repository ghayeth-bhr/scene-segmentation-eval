"""SAM 2 image predictor prompted with boxes (ground truth or detector) -> predictions/sam2.json (COCO RLE)."""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from masks import encode


def gt_boxes(ann_path: str) -> dict[str, list[tuple[list[float], int]]]:
    with open(ann_path, encoding="utf-8") as fh:
        coco = json.load(fh)
    names = {im["id"]: im["file_name"] for im in coco["images"]}
    out = defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        out[names[a["image_id"]]].append(([x, y, x + w, y + h], a["category_id"]))
    return out


def detector_boxes(images: list[str], folder: str, model_path: str, conf: float):
    from ultralytics import YOLO
    model = YOLO(model_path)
    out = defaultdict(list)
    for f in images:
        r = model.predict(os.path.join(folder, f), conf=conf, verbose=False)[0]
        for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            out[f].append((box.tolist(), int(cls)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True, help="COCO json (image list, and boxes when --box-source gt)")
    ap.add_argument("--checkpoint", default="sam2.1_hiera_small.pt")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--box-source", choices=["gt", "detector"], default="gt")
    ap.add_argument("--detector", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="predictions/sam2.json")
    a = ap.parse_args()

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = a.device if torch.cuda.is_available() else "cpu"
    predictor = SAM2ImagePredictor(build_sam2(a.config, a.checkpoint, device=device))

    boxes = gt_boxes(a.ann)
    if a.box_source == "detector":
        boxes = detector_boxes(sorted(boxes), a.images, a.detector, a.conf)

    preds = []
    with torch.inference_mode():
        for fname, items in sorted(boxes.items()):
            if not items:
                continue
            img = cv2.cvtColor(cv2.imread(os.path.join(a.images, fname)), cv2.COLOR_BGR2RGB)
            predictor.set_image(img)
            box_arr = np.array([b for b, _ in items], dtype=np.float32)
            masks, scores, _ = predictor.predict(box=box_arr, multimask_output=False)
            masks = masks.reshape(len(items), *img.shape[:2]) > 0
            scores = np.asarray(scores).reshape(len(items), -1)[:, 0]
            for (box, cls), m, s in zip(items, masks, scores):
                x1, y1, x2, y2 = box
                preds.append({"file_name": fname, "score": float(s), "class_id": cls,
                              "bbox": [x1, y1, x2 - x1, y2 - y1], "segmentation": encode(m)})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"method": f"sam2_{a.box_source}", "model": os.path.basename(a.checkpoint),
                   "predictions": preds}, fh)
    print(f"{len(preds)} masks -> {a.out} (device={device}, boxes={a.box_source})")


if __name__ == "__main__":
    main()
