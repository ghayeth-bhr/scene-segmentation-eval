"""Build a street-scene evaluation set from COCO val2017: images that contain people and cars,
non-crowd polygon masks for those classes only, plus a relative zone polygon (lower-centre region,
like a pavement / crossing zone seen from a fixed camera)."""
import argparse
import json
import os
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

KEEP = {1: "person", 3: "car"}
ZONE = [[0.30, 0.45], [0.70, 0.45], [0.95, 1.0], [0.05, 1.0]]   # trapezoid, relative coordinates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ann", required=True, help="annotations/instances_val2017.json")
    ap.add_argument("--out", default="data/scene")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-area", type=float, default=1000, help="ignore tiny objects (pixels)")
    a = ap.parse_args()

    coco = json.load(open(a.ann, encoding="utf-8"))
    anns = defaultdict(list)
    for x in coco["annotations"]:
        if x["category_id"] in KEEP and not x["iscrowd"] and x["area"] >= a.min_area \
                and isinstance(x["segmentation"], list):
            anns[x["image_id"]].append(x)
    has = lambda iid, c: any(x["category_id"] == c for x in anns[iid])  # noqa: E731
    images = [im for im in sorted(coco["images"], key=lambda im: im["id"]) if has(im["id"], 1) and has(im["id"], 3)]
    images = images[: a.n]
    sub = {"images": images,
           "categories": [c for c in coco["categories"] if c["id"] in KEEP],
           "annotations": [x for im in images for x in anns[im["id"]]]}

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    with open(os.path.join(a.out, "annotations.json"), "w", encoding="utf-8") as fh:
        json.dump(sub, fh)
    with open(os.path.join(a.out, "zone.json"), "w", encoding="utf-8") as fh:
        json.dump({"zone": ZONE}, fh)

    def fetch(im):
        path = os.path.join(a.out, "images", im["file_name"])
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{im['file_name']}", path)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, images))
    print(f"{len(images)} images, {len(sub['annotations'])} objects (person + car) -> {a.out}")


if __name__ == "__main__":
    main()
