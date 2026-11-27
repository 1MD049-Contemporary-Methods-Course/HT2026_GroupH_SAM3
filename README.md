# HT2026 Group H — SAM3

Task 2 of 1MD049 Contemporary Methods in Visual Data Processing (Uppsala University, HT2026).

Zero-shot evaluation of Meta's SAM3 on a COCO val2017 subset: text-only prompts versus text plus one positive box exemplar, including instance counting.

## Layout

| Path | Owner | Content |
| --- | --- | --- |
| `data/` | A | Subset selection (`subset.py`) |
| `evaluation/` | A | Matching and metrics (`metrics.py`) |
| `infer/` | B | SAM3 inference, writes prediction files |
| `analysis/` | C | Threshold sweeps, tables, figures |
| `tests/` | all | `python -m unittest discover tests` |

## Environment (Pelle)

```bash
source /proj/uppmax2026-1-299/groupH_sam3/env.sh
```

Shared weights, COCO data and the Python environment are described in `/proj/uppmax2026-1-299/groupH_sam3/README.md`. Run everything from the repo root.

## Shared subset

`$GROUPH/data/subset_ids.json` (made by `python data/subset.py --out ...`, seed 0): 34 images each of sheep, car and person, 102 entries, 599 GT instances. An image qualifies for a category if it has 3–15 instances of it and none is `iscrowd`. Each entry fixes the box exemplar (`prompt_ann_id`, `prompt_box_xyxy`: the largest GT instance, pixel xyxy). Use these boxes for the text+box prompt.

## Prediction format and evaluation

Inference writes one JSON list per run, one record per (image, category, prompt type):

```json
{"image_id": 5992, "category": "sheep", "prompt_type": "text+box",
 "instances": [{"score": 0.93, "bbox_xyxy": [x1, y1, x2, y2], "rle": {"size": [h, w], "counts": "..."}}]}
```

Encode masks with `evaluation.metrics.encode_mask`. Keep all instances above a low score (e.g. 0.1); the evaluation applies the score threshold.

```bash
python -m evaluation.metrics --preds preds.json --out metrics.json --score-thr 0.5 \
    --subset $GROUPH/data/subset_ids.json   # optional: exclude the exemplar GT and its match
```

In Python: `evaluate(preds, coco, iou_thr=0.5, score_thr=0.0, exclude=prompt_ann_map(subset))`. Matching is one-to-one (Hungarian on mask IoU, TP if IoU ≥ 0.5). The summary per `category|prompt_type` and `all|prompt_type` holds micro P/R/F1, mean IoU of TPs, and count error (pred − GT: mean, MAE, exact rate). Apply the same `exclude` to both prompt types so the columns are comparable.
