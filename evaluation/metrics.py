"""Match SAM3 instance predictions to COCO GT masks and compute metrics.

Prediction file (written by infer/, one JSON list for a whole run):
    [{"image_id": 5992, "category": "sheep", "prompt_type": "text" | "text+box",
      "instances": [{"score": 0.93, "bbox_xyxy": [x1, y1, x2, y2],
                     "rle": {"size": [h, w], "counts": "..."}}]}, ...]
Encode masks with `encode_mask` below. Save every instance above a low score
(e.g. 0.1); `evaluate` applies `score_thr`, so threshold sweeps need no rerun.

Matching: one-to-one Hungarian assignment on mask IoU; a pair counts as a true
positive if IoU >= iou_thr. Crowd annotations are ignored (the subset has none).

Prompt exclusion: the exemplar box makes one GT instance easy to find. With
`exclude` (from `prompt_ann_map(subset)`), that GT is dropped from the counts,
and so is the prediction matched to it, if any. Use the same `exclude` for both
prompt types so the columns stay comparable.

Usage:
    python -m evaluation.metrics --preds run/preds.json --out run/metrics.json \
        [--subset $GROUPH/data/subset_ids.json]   # --subset enables exclusion
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment


def encode_mask(mask):
    """(H, W) bool array -> JSON-serialisable COCO RLE."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def prompt_ann_map(subset):
    """{(image_id, category): prompt_ann_id} from a loaded subset_ids.json."""
    return {(e["image_id"], e["category"]): e["prompt_ann_id"] for e in subset["entries"]}


def match_record(pred_rles, gt_rles, iou_thr):
    """Return [(pred_idx, gt_idx, iou)] for matched pairs with iou >= iou_thr."""
    if not pred_rles or not gt_rles:
        return []
    iou = np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)))
    rows, cols = linear_sum_assignment(-iou)
    return [(r, c, float(iou[r, c])) for r, c in zip(rows, cols) if iou[r, c] >= iou_thr]


def evaluate_record(record, coco, iou_thr=0.5, score_thr=0.0, exclude=None):
    cat_id = coco.getCatIds(catNms=[record["category"]])[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=record["image_id"], catIds=[cat_id], iscrowd=False))
    gt_rles = [coco.annToRLE(a) for a in anns]
    preds = [p for p in record["instances"] if p["score"] >= score_thr]
    matches = match_record([p["rle"] for p in preds], gt_rles, iou_thr)

    n_gt, n_pred = len(anns), len(preds)
    excluded_ann = None
    if exclude is not None:
        excluded_ann = exclude[(record["image_id"], record["category"])]
        g0 = next(i for i, a in enumerate(anns) if a["id"] == excluded_ann)
        n_gt -= 1
        if any(g == g0 for _, g, _ in matches):
            n_pred -= 1
            matches = [m for m in matches if m[1] != g0]

    return {
        "image_id": record["image_id"],
        "category": record["category"],
        "prompt_type": record["prompt_type"],
        "excluded_ann_id": excluded_ann,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": len(matches),
        "matched_ious": [m[2] for m in matches],
        "count_error": n_pred - n_gt,
    }


def _safe_div(a, b):
    return a / b if b else 0.0


def summarize(rows):
    n_gt = sum(r["n_gt"] for r in rows)
    n_pred = sum(r["n_pred"] for r in rows)
    tp = sum(r["tp"] for r in rows)
    ious = [i for r in rows for i in r["matched_ious"]]
    errors = np.array([r["count_error"] for r in rows])
    precision, recall = _safe_div(tp, n_pred), _safe_div(tp, n_gt)
    return {
        "n_images": len(rows),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "matched_miou": float(np.mean(ious)) if ious else 0.0,
        "count_error_mean": float(errors.mean()),
        "count_mae": float(np.abs(errors).mean()),
        "count_exact_rate": float((errors == 0).mean()),
    }


def evaluate(preds, coco, iou_thr=0.5, score_thr=0.0, exclude=None):
    """Return {"per_image": [...], "summary": {"<category>|<prompt_type>": {...}}}.

    Summary keys include "all|<prompt_type>" pooled over categories. P/R/F1 are
    micro-averaged over instances; matched_miou averages IoU of true positives.
    """
    rows = [evaluate_record(r, coco, iou_thr, score_thr, exclude) for r in preds]
    groups = defaultdict(list)
    for r in rows:
        groups[f"{r['category']}|{r['prompt_type']}"].append(r)
        groups[f"all|{r['prompt_type']}"].append(r)
    return {
        "settings": {"iou_thr": iou_thr, "score_thr": score_thr, "exclude_prompt": exclude is not None},
        "per_image": rows,
        "summary": {k: summarize(v) for k, v in sorted(groups.items())},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default=None, help="subset_ids.json; if given, exclude the exemplar GT")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--annotations", default=None,
                    help="instances JSON (default: $COCO_ROOT/annotations/instances_val2017.json)")
    args = ap.parse_args()

    ann_file = args.annotations or os.path.join(os.environ["COCO_ROOT"], "annotations/instances_val2017.json")
    coco = COCO(ann_file)
    with open(args.preds) as f:
        preds = json.load(f)
    exclude = None
    if args.subset:
        with open(args.subset) as f:
            exclude = prompt_ann_map(json.load(f))
    result = evaluate(preds, coco, args.iou_thr, args.score_thr, exclude)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    for key, s in result["summary"].items():
        print(f"{key:20s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} "
              f"mIoU={s['matched_miou']:.3f} countMAE={s['count_mae']:.2f}")


if __name__ == "__main__":
    main()
