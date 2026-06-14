"""K-fold cross-validated threshold calibration.

Splits stored predictions into k folds. For each fold, calibrates per-class
thresholds on the remaining k-1 folds, then evaluates on the held-out fold.
No image is ever in its own calibration set.

Outputs:
  - Out-of-fold (OOF) metrics: unbiased performance estimate
  - Mean thresholds across folds: ready to paste into lig_vision.py
  - Comparison against native model thresholds
"""
import json, random
import numpy as np
from sklearn.metrics import f1_score, average_precision_score

RESULTS_PATH  = "results/lig_vision.json"
TAXONOMY_PATH = "taxonomy.json"
SEED          = 42
K             = 5
GRID          = [round(i * 0.01, 2) for i in range(1, 100)]
MIN_CAL_POS   = 5   # minimum positives in cal fold to attempt calibration

NATIVE_KEY = {
    "alicia_keys":    0.25,  "boho":           0.325, "box_braids":     0.275,
    "butterfly_locs": 0.5,   "faux_locs":      0.35,  "flat_twist":     0.25,
    "fulani_braids":  0.3,   "invisible_locs": 0.45,  "island_twist":   0.475,
    "lemonade_braids":0.45,  "marley_twists":  0.375, "microlocs":      0.2,
    "mini_twists":    0.25,  "passion_twists": 0.425, "ponytails":      0.25,
    "puff":           0.425, "soft_locs":      0.225, "space_buns":     0.375,
}
NATIVE_BASE = {
    "braids": 0.525, "cornrows": 0.375, "crochet": 0.425,
    "locs":   0.35,  "twist":    0.425,
}


def _best_threshold(cal_preds, head, cls, fallback):
    gt    = [1 if cls in p.get(f"{head}_gt", []) else 0 for p in cal_preds]
    probs = [p.get(f"{head}_probs", {}).get(cls, 0.0)   for p in cal_preds]
    if sum(gt) < MIN_CAL_POS:
        return fallback
    best_t, best_f1 = fallback, -1.0
    for t in GRID:
        preds = [1 if pr >= t else 0 for pr in probs]
        f1 = f1_score(gt, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def _eval_head(predictions, head, classes, thresholds):
    head_preds = [p for p in predictions if p.get(f"{head}_gt")]
    per_class  = {}
    for cls in classes:
        gt    = [1 if cls in p[f"{head}_gt"] else 0 for p in head_preds]
        probs = [p.get(f"{head}_probs", {}).get(cls, 0.0) for p in head_preds]
        if sum(gt) == 0:
            continue
        preds = [1 if pr >= thresholds[cls] else 0 for pr in probs]
        ap = float(average_precision_score(gt, probs)) if len(set(gt)) > 1 else None
        per_class[cls] = {
            "f1":      round(float(f1_score(gt, preds, zero_division=0)), 4),
            "ap":      round(ap, 4) if ap else None,
            "support": int(sum(gt)),
        }
    macro = round(float(np.mean([v["f1"] for v in per_class.values()])), 4) if per_class else 0.0
    return macro, per_class


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)

    preds = data["predictions"]
    random.Random(SEED).shuffle(preds)
    folds = [preds[i::K] for i in range(K)]

    print(f"Total predictions : {len(preds)}")
    print(f"Folds             : {K}  (~{len(preds)//K} images each)\n")

    fold_key_thresholds  = {cls: [] for cls in taxonomy["key"]}
    fold_base_thresholds = {cls: [] for cls in taxonomy["base"]}
    oof_key_preds  = []
    oof_base_preds = []

    for k in range(K):
        cal   = [p for i, fold in enumerate(folds) if i != k for p in fold]
        held  = folds[k]

        key_t  = {cls: _best_threshold(cal, "key",  cls, NATIVE_KEY.get(cls, 0.3))
                  for cls in taxonomy["key"]}
        base_t = {cls: _best_threshold(cal, "base", cls, NATIVE_BASE.get(cls, 0.3))
                  for cls in taxonomy["base"]}

        for cls in taxonomy["key"]:
            fold_key_thresholds[cls].append(key_t[cls])
        for cls in taxonomy["base"]:
            fold_base_thresholds[cls].append(base_t[cls])

        for p in held:
            oof_key_preds.append({"p": p, "thresholds": key_t})
            oof_base_preds.append({"p": p, "thresholds": base_t})

    mean_key  = {cls: round(float(np.mean(ts)), 2) for cls, ts in fold_key_thresholds.items()}
    mean_base = {cls: round(float(np.mean(ts)), 2) for cls, ts in fold_base_thresholds.items()}

    # OOF evaluation: use each image's fold-specific threshold
    def oof_eval(oof_records, head, classes):
        head_records = [r for r in oof_records if r["p"].get(f"{head}_gt")]
        per_class = {}
        for cls in classes:
            gt, preds_bin, probs = [], [], []
            for r in head_records:
                p   = r["p"]
                thr = r["thresholds"][cls]
                g   = 1 if cls in p.get(f"{head}_gt", []) else 0
                pr  = p.get(f"{head}_probs", {}).get(cls, 0.0)
                gt.append(g); probs.append(pr)
                preds_bin.append(1 if pr >= thr else 0)
            if sum(gt) == 0:
                continue
            ap = float(average_precision_score(gt, probs)) if len(set(gt)) > 1 else None
            per_class[cls] = {
                "f1":      round(float(f1_score(gt, preds_bin, zero_division=0)), 4),
                "ap":      round(ap, 4) if ap else None,
                "support": int(sum(gt)),
            }
        macro = round(float(np.mean([v["f1"] for v in per_class.values()])), 4) if per_class else 0.0
        return macro, per_class

    oof_key_f1,  oof_key_pc  = oof_eval(oof_key_preds,  "key",  taxonomy["key"])
    oof_base_f1, oof_base_pc = oof_eval(oof_base_preds, "base", taxonomy["base"])

    nat_key_f1,  nat_key_pc  = _eval_head(preds, "key",  taxonomy["key"],  NATIVE_KEY)
    nat_base_f1, nat_base_pc = _eval_head(preds, "base", taxonomy["base"], NATIVE_BASE)

    print("=== MACRO F1 COMPARISON ===")
    print(f"{'':30} {'native':>8} {'OOF (CV)':>10}")
    print(f"  key  macro F1                  {nat_key_f1:>8.4f} {oof_key_f1:>10.4f}")
    print(f"  base macro F1                  {nat_base_f1:>8.4f} {oof_base_f1:>10.4f}")

    print(f"\n=== KEY PER-CLASS ===")
    print(f"  {'class':<20} {'nat_t':>6} {'cv_t':>6} {'native_f1':>10} {'oof_f1':>8}  {'ap':>7}  sup")
    for cls in taxonomy["key"]:
        n = nat_key_pc.get(cls, {})
        o = oof_key_pc.get(cls, {})
        if not n and not o:
            continue
        diff   = o.get("f1", 0) - n.get("f1", 0)
        marker = " +" if diff > 0.02 else (" -" if diff < -0.02 else "")
        print(f"  {cls:<20} {NATIVE_KEY.get(cls,0):>6.3f} {mean_key.get(cls,0):>6.2f} "
              f"{n.get('f1',0):>10.4f} {o.get('f1',0):>8.4f}  "
              f"{str(o.get('ap','-')):>7}  {o.get('support',0)}{marker}")

    print(f"\n=== BASE PER-CLASS ===")
    print(f"  {'class':<12} {'nat_t':>6} {'cv_t':>6} {'native_f1':>10} {'oof_f1':>8}  {'ap':>7}  sup")
    for cls in taxonomy["base"]:
        n = nat_base_pc.get(cls, {})
        o = oof_base_pc.get(cls, {})
        if not n and not o:
            continue
        diff   = o.get("f1", 0) - n.get("f1", 0)
        marker = " +" if diff > 0.02 else (" -" if diff < -0.02 else "")
        print(f"  {cls:<12} {NATIVE_BASE.get(cls,0):>6.3f} {mean_base.get(cls,0):>6.2f} "
              f"{n.get('f1',0):>10.4f} {o.get('f1',0):>8.4f}  "
              f"{str(o.get('ap','-')):>7}  {o.get('support',0)}{marker}")

    print("\n=== MEAN CV THRESHOLDS ===")
    print("\n_KEY_THRESHOLDS = {")
    for cls, t in sorted(mean_key.items()):
        print(f'    "{cls}":{" " * (20 - len(cls))}{t},')
    print("}")
    print("\n_BASE_THRESHOLDS = {")
    for cls, t in sorted(mean_base.items()):
        print(f'    "{cls}":{" " * (10 - len(cls))}{t},')
    print("}")

    # write OOF metrics back to lig_vision.json
    data["metrics"]["key"]["macro_f1"]   = oof_key_f1
    data["metrics"]["key"]["per_class"]  = oof_key_pc
    data["metrics"]["base"]["macro_f1"]  = oof_base_f1
    data["metrics"]["base"]["per_class"] = oof_base_pc
    data["calibration"] = {
        "method": "5-fold cross-validation",
        "description": (
            "Per-class thresholds are the mean of fold-specific optima calibrated on k-1 folds. "
            "Reported F1 metrics are out-of-fold (OOF): each image is evaluated using thresholds "
            "calibrated on the other four folds, so no image appears in its own calibration set. "
            "mAP is threshold-independent and is computed on all predictions."
        ),
        "k": K,
        "seed": SEED,
        "min_cal_positives": MIN_CAL_POS,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nresults/lig_vision.json updated with OOF metrics.")


if __name__ == "__main__":
    main()
