"""
Hairstyle classification benchmark.

Usage:
    python evaluate.py [--models MODEL [MODEL ...]] [--limit N]

Examples:
    python evaluate.py
    python evaluate.py --models lig_vision gpt
    python evaluate.py --limit 50
"""

import argparse
import json
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

import dataset
import metrics
from models import lig_vision, gpt, gemini, claude_model

load_dotenv()

TAXONOMY_PATH = Path("taxonomy.json")
RESULTS_DIR   = Path("results")

ALL_MODELS = {
    "lig_vision": lig_vision.classify,
    "gpt":        gpt.classify,
    "gemini":     gemini.classify,
    "claude":     claude_model.classify,
}


def _is_complete(name: str) -> bool:
    p = RESULTS_DIR / f"{name}.json"
    if not p.exists():
        return False
    with open(p) as f:
        r = json.load(f)
    return r.get("n_errors", 1) == 0


def run(model_names: list, limit: int | None) -> None:
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)

    RESULTS_DIR.mkdir(exist_ok=True)

    pending = [n for n in model_names if not _is_complete(n)]
    skipped = [n for n in model_names if _is_complete(n)]

    for name in skipped:
        print(f"\n--- {name} (skipped — complete) ---")

    records: list = []
    if pending:
        records = dataset.load_records(taxonomy)
        print(f"Loaded {len(records)} labelled images")
        if limit:
            records = records[:limit]
            print(f"Limited to {limit} images")

    summary_rows = []

    for name in skipped:
        with open(RESULTS_DIR / f"{name}.json") as f:
            r = json.load(f)
        m = r["metrics"]
        summary_rows.append({
            "model":    name,
            "key_f1":   m["key"]["macro_f1"],
            "key_map":  m["key"].get("map"),
            "base_f1":  m["base"]["macro_f1"],
            "n_images": r["n_images"],
            "n_errors": r["n_errors"],
        })

    for name in pending:
        classify_fn = ALL_MODELS[name]
        print(f"\n--- {name} ---")

        predictions = []
        errors      = 0

        for i, rec in enumerate(records, 1):
            try:
                pred = classify_fn(rec["img_bytes"], taxonomy)
            except Exception as exc:
                print(f"  [{i}/{len(records)}] ERROR: {exc}")
                errors += 1
                continue

            predictions.append({
                "filename":   rec["filename"],
                "key_gt":     rec["key"],
                "base_gt":    rec["base"],
                "key_pred":   pred["key"],
                "base_pred":  pred["base"],
                "key_probs":  pred.get("key_probs"),
                "base_probs": pred.get("base_probs"),
                "latency_ms": pred.get("latency_ms"),
            })

            if i % 10 == 0:
                print(f"  {i}/{len(records)} done  ({errors} errors)")

        m = metrics.compute(predictions, taxonomy)

        out = {
            "model":        name,
            "evaluated_at": str(date.today()),
            "n_images":     len(predictions),
            "n_errors":     errors,
            "metrics":      m,
            "predictions":  predictions,
        }

        out_path = RESULTS_DIR / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        key_f1  = m["key"]["macro_f1"]
        base_f1 = m["base"]["macro_f1"]
        key_map = m["key"].get("map")
        print(f"  key  macro-F1={key_f1:.3f}  mAP={key_map or 'n/a'}")
        print(f"  base macro-F1={base_f1:.3f}")
        print(f"  Saved -> {out_path}")

        summary_rows.append({
            "model":    name,
            "key_f1":   key_f1,
            "key_map":  key_map,
            "base_f1":  base_f1,
            "n_images": len(predictions),
            "n_errors": errors,
        })

    print("\n=== Summary ===")
    print(f"{'Model':<15} {'Key F1':>8} {'Key mAP':>9} {'Base F1':>9} {'Images':>8}")
    print("-" * 55)
    for row in summary_rows:
        map_str = f"{row['key_map']:.3f}" if row["key_map"] else "  n/a"
        print(
            f"{row['model']:<15}"
            f" {row['key_f1']:>8.3f}"
            f" {map_str:>9}"
            f" {row['base_f1']:>9.3f}"
            f" {row['n_images']:>8}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=list(ALL_MODELS.keys()),
        choices=list(ALL_MODELS.keys()),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.models, args.limit)
