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
  zone.json          {"zone": [[x1, y1], [x2, y2], ...]}   zone polygon, in pixels or relative (0-1) coordinates
```

## Usage

```bash
# 1. Predictions (each writes predictions/<method>.json in COCO-RLE format)
python tools/make_coco_scene.py --ann instances_val2017.json --out data/scene   # or your own scene
python predict_light.py --images data/scene/images --model yolo11n-seg.pt --classes 0 2
python predict_sam2.py  --images data/scene/images --ann data/scene/annotations.json \
                        --checkpoint sam2.1_hiera_small.pt --box-source gt --out predictions/sam2_gt.json
python predict_sam2.py  --images data/scene/images --ann data/scene/annotations.json                         --checkpoint sam2.1_hiera_small.pt --box-source detector --classes 0 2                         --out predictions/sam2_detector.json

# 2. Evaluation: mask IoU + zone-decision agreement, bbox baseline included automatically
python evaluate.py --ann data/scene/annotations.json --zone data/scene/zone.json \
                   --pred predictions/yolo_seg.json predictions/sam2_gt.json predictions/sam2_detector.json

# 3. Visual check of disagreements
python visualize.py --images data/scene/images --ann data/scene/annotations.json \
                    --zone data/scene/zone.json --pred predictions/sam2_gt.json --out viz/
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

**Setup:** 100 street scenes from COCO val2017 containing both people and cars
(`tools/make_coco_scene.py`), 490 ground-truth objects (person + car, non-crowd, area ≥ 1000 px).
Zone: a trapezoid over the lower-centre of each frame in relative coordinates
`[[0.30,0.45],[0.70,0.45],[0.95,1.0],[0.05,1.0]]`, decision threshold 0.5.
YOLO11n-seg (Ultralytics) and SAM 2.1 Hiera-S, both on an RTX 4050 Laptop GPU (PyTorch 2.14, CUDA 12.6).
Timing = median per image (SAM 2: image encoding + mask decoding for all boxes of the image).

| Method | Objects | Mean mask IoU | Zone decision accuracy | Decisions changed vs. bbox | ms / image |
|---|---|---|---|---|---|
| bbox-only (GT boxes) | 490 | 0.572 | 0.953 | — | — |
| YOLO11n-seg | 490 | 0.656 | 0.943 | 37 | 37 |
| SAM 2 + GT boxes | 490 | **0.820** | **0.990** | 26 | 168 |
| SAM 2 + YOLO11n boxes | 490 | 0.666 | 0.943 | 43 | 173 (+ detector) |

**What the numbers say**

- **A box is a poor mask** (IoU 0.57) but a **good zone signal**: box-based zone logic already
  takes the right in-zone decision for 95% of objects.
- **SAM 2 fixes most of the remaining errors when it gets good boxes** (99.0%, IoU 0.82),
  at ~4.5x the cost of YOLO11n-seg (26 decisions change compared with boxes; inspect them
  with `visualize.py`).
- **In a full pipeline, the detector is the bottleneck**: prompted with YOLO11n boxes, SAM 2
  drops to the same zone accuracy as YOLO11n-seg (94.3%), because missed or merged detections
  cannot be repaired by better masks.
- Practical conclusion for an edge video-analytics system: keep box-based zone rules as the
  default, and spend compute on segmentation only for borderline objects (box fraction in zone
  close to the threshold), where it changes the decision.

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
