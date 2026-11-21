"""Mask helpers: polygon / box / RLE rasterisation, IoU, fraction of an object inside a zone."""
from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def polygon_to_mask(polygons: list[list[float]], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def box_to_mask(box_xywh: list[float], h: int, w: int) -> np.ndarray:
    x, y, bw, bh = box_xywh
    m = np.zeros((h, w), dtype=bool)
    m[max(0, int(y)):min(h, int(round(y + bh))), max(0, int(x)):min(w, int(round(x + bw)))] = True
    return m


def ann_to_mask(ann: dict, h: int, w: int) -> np.ndarray:
    seg = ann["segmentation"]
    if isinstance(seg, list):
        return polygon_to_mask(seg, h, w)
    if isinstance(seg["counts"], list):                      # uncompressed RLE
        seg = mask_utils.frPyObjects(seg, h, w)
    return mask_utils.decode(seg).astype(bool)


def encode(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def decode(rle: dict) -> np.ndarray:
    return mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def zone_pixels(zone: list[list[float]], h: int, w: int) -> list[list[float]]:
    """Zones may be given in pixels or in relative coordinates (all values in [0, 1])."""
    if all(0 <= v <= 1 for pt in zone for v in pt):
        return [[x * w, y * h] for x, y in zone]
    return zone


def zone_mask(zone: list[list[float]], h: int, w: int) -> np.ndarray:
    return polygon_to_mask([sum(zone_pixels(zone, h, w), [])], h, w)


def fraction_in_zone(obj: np.ndarray, zone: np.ndarray) -> float:
    area = obj.sum()
    return float(np.logical_and(obj, zone).sum() / area) if area else 0.0
