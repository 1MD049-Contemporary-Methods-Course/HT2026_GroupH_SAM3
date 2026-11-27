"""Matching and metrics on a hand-built 40x40 COCO image with three square sheep."""
import unittest

import numpy as np
from pycocotools.coco import COCO

from evaluation.metrics import encode_mask, evaluate, match_record

H = W = 40
# (x, y, size) of three non-overlapping GT squares; ann ids 1, 2, 3.
SQUARES = [(0, 0, 10), (20, 0, 10), (0, 20, 16)]


def square_mask(x, y, s):
    m = np.zeros((H, W), bool)
    m[y:y + s, x:x + s] = True
    return m


def make_coco():
    anns = []
    for i, (x, y, s) in enumerate(SQUARES, start=1):
        anns.append({"id": i, "image_id": 1, "category_id": 20, "iscrowd": 0, "area": s * s,
                     "bbox": [x, y, s, s], "segmentation": encode_mask(square_mask(x, y, s))})
    coco = COCO()
    coco.dataset = {"images": [{"id": 1, "height": H, "width": W, "file_name": "x.jpg"}],
                    "categories": [{"id": 20, "name": "sheep"}], "annotations": anns}
    coco.createIndex()
    return coco


def inst(mask, score):
    return {"score": score, "bbox_xyxy": [0, 0, 1, 1], "rle": encode_mask(mask)}


def record(instances, prompt_type="text"):
    return {"image_id": 1, "category": "sheep", "prompt_type": prompt_type, "instances": instances}


class MetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coco = make_coco()

    def test_rle_roundtrip_is_json_safe(self):
        rle = encode_mask(square_mask(*SQUARES[0]))
        self.assertIsInstance(rle["counts"], str)
        self.assertEqual(rle["size"], [H, W])

    def test_perfect_predictions(self):
        preds = [record([inst(square_mask(*sq), 0.9) for sq in SQUARES])]
        s = evaluate(preds, self.coco)["summary"]["sheep|text"]
        self.assertEqual((s["tp"], s["n_pred"], s["n_gt"]), (3, 3, 3))
        self.assertAlmostEqual(s["f1"], 1.0)
        self.assertAlmostEqual(s["matched_miou"], 1.0)
        self.assertEqual(s["count_mae"], 0)

    def test_false_positive_duplicate_and_score_filter(self):
        preds = [record([
            inst(square_mask(*SQUARES[0]), 0.9),
            inst(square_mask(*SQUARES[0]), 0.8),   # duplicate: only one can match
            inst(square_mask(30, 30, 8), 0.7),     # background false positive
            inst(square_mask(*SQUARES[1]), 0.2),   # true, but below score_thr
        ])]
        row = evaluate(preds, self.coco, score_thr=0.5)["per_image"][0]
        self.assertEqual((row["tp"], row["n_pred"], row["n_gt"]), (1, 3, 3))
        self.assertEqual(row["count_error"], 0)  # right count, wrong instances
        row = evaluate(preds, self.coco, score_thr=0.1)["per_image"][0]
        self.assertEqual((row["tp"], row["n_pred"]), (2, 4))

    def test_iou_threshold(self):
        x, y, s = SQUARES[2]
        half = square_mask(x, y, s)
        half[:, x + s // 2:] = False  # IoU 0.5 with the GT square
        rows = evaluate([record([inst(half, 0.9)])], self.coco, iou_thr=0.5)["per_image"]
        self.assertEqual(rows[0]["tp"], 1)
        rows = evaluate([record([inst(half, 0.9)])], self.coco, iou_thr=0.6)["per_image"]
        self.assertEqual(rows[0]["tp"], 0)

    def test_hungarian_prefers_best_overall_assignment(self):
        gt = [encode_mask(square_mask(*sq)) for sq in SQUARES[:2]]
        both = square_mask(*SQUARES[0]) | square_mask(*SQUARES[1])
        pairs = match_record([encode_mask(both), gt[0]], gt, iou_thr=0.4)
        self.assertEqual(sorted((p, g) for p, g, _ in pairs), [(0, 1), (1, 0)])

    def test_exclude_prompt_instance(self):
        exclude = {(1, "sheep"): 3}
        found = record([inst(square_mask(*sq), 0.9) for sq in SQUARES], "text+box")
        row = evaluate([found], self.coco, exclude=exclude)["per_image"][0]
        self.assertEqual((row["tp"], row["n_pred"], row["n_gt"]), (2, 2, 2))
        # Exemplar missed: its GT is still dropped, unmatched predictions stay.
        missed = record([inst(square_mask(*SQUARES[0]), 0.9), inst(square_mask(30, 30, 8), 0.9)])
        row = evaluate([missed], self.coco, exclude=exclude)["per_image"][0]
        self.assertEqual((row["tp"], row["n_pred"], row["n_gt"]), (1, 2, 2))

    def test_summary_groups_and_empty_predictions(self):
        preds = [record([inst(square_mask(*SQUARES[0]), 0.9)], "text"), record([], "text+box")]
        summary = evaluate(preds, self.coco)["summary"]
        self.assertEqual(set(summary), {"sheep|text", "sheep|text+box", "all|text", "all|text+box"})
        empty = summary["sheep|text+box"]
        self.assertEqual((empty["precision"], empty["recall"], empty["count_error_mean"]), (0.0, 0.0, -3.0))


if __name__ == "__main__":
    unittest.main()
