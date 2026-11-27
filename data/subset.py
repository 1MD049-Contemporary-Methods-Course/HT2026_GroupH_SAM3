"""Select the shared COCO val2017 evaluation subset.

Rule, per category: keep an image if it has 3-15 instances of that category and
none of them is iscrowd=1; then draw a fixed-seed random sample. Each entry
also fixes the positive box exemplar (the largest GT instance), so every
experiment prompts with the same box and the evaluation can exclude it.

Usage (repo root, after sourcing env.sh):
    python data/subset.py --out $GROUPH/data/subset_ids.json
"""
import argparse
import json
import os
import random
from pathlib import Path

from pycocotools.coco import COCO

DEFAULT_CATEGORIES = ["sheep", "car", "person"]


def eligible_images(coco, cat_id, min_inst, max_inst):
    """Image ids with min_inst..max_inst instances of cat_id and no crowd annotation."""
    keep = []
    for img_id in sorted(coco.getImgIds(catIds=[cat_id])):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=None))
        if any(a["iscrowd"] for a in anns) or not min_inst <= len(anns) <= max_inst:
            continue
        keep.append(img_id)
    return keep


def build_subset(coco, categories, per_category, seed, min_inst=3, max_inst=15):
    entries = []
    for name in categories:
        cat_id = coco.getCatIds(catNms=[name])[0]
        candidates = eligible_images(coco, cat_id, min_inst, max_inst)
        if len(candidates) < per_category:
            raise ValueError(f"{name}: only {len(candidates)} eligible images, need {per_category}")
        # Seed per category so changing the category list does not reshuffle the others.
        chosen = sorted(random.Random(f"{seed}-{name}").sample(candidates, per_category))
        for img_id in chosen:
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=False))
            exemplar = max(anns, key=lambda a: a["area"])
            x, y, w, h = exemplar["bbox"]
            entries.append({
                "image_id": img_id,
                "file_name": coco.loadImgs(img_id)[0]["file_name"],
                "category": name,
                "category_id": cat_id,
                "n_gt": len(anns),
                "prompt_ann_id": exemplar["id"],
                "prompt_box_xyxy": [x, y, x + w, y + h],
            })
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--per-category", type=int, default=34)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--annotations", default=None,
                    help="instances JSON (default: $COCO_ROOT/annotations/instances_val2017.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ann_file = args.annotations or os.path.join(os.environ["COCO_ROOT"], "annotations/instances_val2017.json")
    coco = COCO(ann_file)
    entries = build_subset(coco, args.categories, args.per_category, args.seed)
    subset = {
        "annotations": "instances_val2017.json",
        "rule": {"min_instances": 3, "max_instances": 15, "exclude_iscrowd_images": True,
                 "exemplar": "largest GT instance by area"},
        "categories": args.categories,
        "per_category": args.per_category,
        "seed": args.seed,
        "entries": entries,
    }
    Path(args.out).write_text(json.dumps(subset, indent=1))
    print(f"wrote {len(entries)} entries to {args.out}")


if __name__ == "__main__":
    main()
