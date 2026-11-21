# Semantic Segmentation for Scene Understanding — masks vs. bounding boxes for zone logic

Does a video-analytics system that decides "is this object inside the zone?" from
**bounding boxes** make the same decisions as one that uses **segmentation masks**?

This repo applies **SAM 2** (prompted with boxes) and a **lightweight segmentation model**
(YOLO-seg) to a real-world scene, evaluates both with **IoU** against ground-truth masks,
and compares them with the **bounding-box-only zone logic** commonly used in production.

```
image + GT masks + zone polygon
        │
        ├── bbox-only baseline ─────── box rasterised as a mask
        ├── YOLO-seg (lightweight) ─── predicted instance masks
        └── SAM 2 (box prompts) ────── refined masks from the same boxes
                    │
             evaluate.py
        ├── mask IoU per object / per method
        └── zone decision agreement (in-zone yes/no) vs. ground truth
```

## Why

Box-based zone rules (e.g. "fraction of the box inside the polygon > 0.5") are cheap and
robust, but a box over-covers thin, diagonal or partially occluded objects. Masks are
more precise but cost more compute. The useful question is not "which mask is
prettiest" but **how often the zone decision changes**, and whether that justifies the
extra latency on an edge device.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# SAM 2 (Meta): pip install git+https://github.com/facebookresearch/sam2.git
```

## Data format

```
data/scene/
  images/            frame_0001.jpg ...
  annotations.json   COCO instances with polygon segmentation (e.g. exported from CVAT / Roboflow)
  zone.json          {"zone": [[x1, y1], [x2, y2], ...]}   zone polygon in pixels
```

## Usage

```bash
# 1. Predictions (each writes predictions/<method>.json in COCO-RLE format)
python predict_light.py --images data/scene/images --model yolo11n-seg.pt
python predict_sam2.py  --images data/scene/images --ann data/scene/annotations.json \
                        --checkpoint sam2.1_hiera_small.pt --config configs/sam2.1/sam2.1_hiera_s.yaml

# 2. Evaluation: mask IoU + zone-decision agreement, bbox baseline included automatically
python evaluate.py --ann data/scene/annotations.json --zone data/scene/zone.json \
                   --pred predictions/yolo_seg.json predictions/sam2.json

# 3. Visual check of disagreements
python visualize.py --images data/scene/images --ann data/scene/annotations.json \
                    --zone data/scene/zone.json --pred predictions/sam2.json --out viz/
```

SAM 2 is prompted with the ground-truth boxes by default (`--box-source gt`) to isolate
mask quality from detection quality; use `--box-source detector` to prompt it with YOLO
boxes and measure the full pipeline.

## Metrics

- **Mask IoU** between each ground-truth object and its matched prediction
  (Hungarian matching on IoU, unmatched objects count as IoU 0).
- **Zone decision**: an object is "in zone" when the fraction of its area inside the
  zone polygon exceeds `--zone-thr` (default 0.5). The bbox baseline uses the box area,
  the other methods use mask area, the reference uses the ground-truth mask.
  Reported as accuracy and the number of flipped decisions vs. the bbox baseline.

## Results

Fill after running `evaluate.py` (it writes `results/summary.md`).

| Method | Mean mask IoU | Zone decision accuracy | Decisions changed vs. bbox | Notes |
|---|---|---|---|---|
| bbox-only | | | — | |
| YOLO-seg | | | | |
| SAM 2 | | | | |

## Project layout

```
predict_light.py   lightweight instance segmentation (Ultralytics YOLO-seg)
predict_sam2.py    SAM 2 image predictor prompted with boxes
evaluate.py        IoU matching + zone decision agreement, writes results/summary.md
visualize.py       overlays GT / prediction / zone for disagreement cases
masks.py           RLE / polygon / box rasterisation, IoU, zone overlap helpers
```

## References

- [SAM 2 — Segment Anything in Images and Videos (Meta)](https://github.com/facebookresearch/sam2)
- [Ultralytics YOLO segmentation](https://docs.ultralytics.com/tasks/segment/)
- [pycocotools](https://github.com/cocodataset/cocoapi) for RLE encoding
- [Roboflow Supervision](https://github.com/roboflow/supervision) — polygon zone idea

## License

MIT © Ghayeth Ben Haj Rhouma
