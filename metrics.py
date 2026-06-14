"""Compute per-class and macro-averaged F1, plus mAP where probabilities are available."""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def compute(predictions: list, taxonomy: dict) -> dict:
    """
    predictions: list of dicts with keys:
      key_pred, base_pred      — active label lists
      key_probs, base_probs    — {class: prob} or None
      key_gt, base_gt          — ground-truth label lists

    Returns per-head metrics dict.
    """
    results = {}

    for head in ("key", "base"):
        # only score images that have at least one GT label for this head
        head_preds = [p for p in predictions if p[f"{head}_gt"]]
        classes    = taxonomy[head]
        n          = len(head_preds)

        results[f"n_{head}_images"] = n

        gt_mat   = np.zeros((n, len(classes)), dtype=int)
        pred_mat = np.zeros((n, len(classes)), dtype=int)
        prob_mat = np.full((n, len(classes)), np.nan)

        for i, p in enumerate(head_preds):
            for j, cls in enumerate(classes):
                if cls in p[f"{head}_gt"]:
                    gt_mat[i, j] = 1
                if cls in p[f"{head}_pred"]:
                    pred_mat[i, j] = 1
                probs = p.get(f"{head}_probs")
                if probs and cls in probs:
                    prob_mat[i, j] = probs[cls]

        per_class = {}
        for j, cls in enumerate(classes):
            if gt_mat[:, j].sum() == 0:
                continue
            f1  = f1_score(gt_mat[:, j], pred_mat[:, j], zero_division=0)
            ap  = None
            if not np.isnan(prob_mat[:, j]).all():
                ap = round(float(average_precision_score(gt_mat[:, j], prob_mat[:, j])), 4)
            per_class[cls] = {
                "f1":      round(float(f1), 4),
                "ap":      ap,
                "support": int(gt_mat[:, j].sum()),
            }

        valid_f1s = [v["f1"] for v in per_class.values()]
        valid_aps = [v["ap"] for v in per_class.values() if v["ap"] is not None]

        results[head] = {
            "macro_f1": round(float(np.mean(valid_f1s)), 4) if valid_f1s else 0.0,
            "map":      round(float(np.mean(valid_aps)), 4) if valid_aps else None,
            "per_class": per_class,
        }

    return results
