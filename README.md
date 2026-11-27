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
