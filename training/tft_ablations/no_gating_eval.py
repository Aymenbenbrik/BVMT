"""training/tft_ablations/no_gating_eval.py

Post-hoc TFT confidence-gating ablation. Reads a per-row TFT v3
prediction CSV (produced by training/evaluate_tft_v3_per_row.py) and
sweeps the confidence threshold in Eq. 3 of the article. For each
threshold, abstain on low-confidence rows and report (coverage,
accuracy on covered rows). The "no gating" reference is threshold=0
(all rows kept) which equals the headline 77.4 %.

Why post-hoc and not a separate training: the confidence-gating
mechanism (Eq. 3) acts on the OUTPUT of the TFT (it derives a
confidence score from quantile width per agents/technical_agent.py
line ~449 and abstains/down-weights low-confidence rows). It does NOT
change the training objective, so a fresh training run would produce
the same checkpoint. The ablation is therefore an evaluation-time
sweep over the confidence threshold tau in [0.0, 0.5, 0.55, 0.60, 0.65,
0.70, 0.75].

This script is the only Action 1.4 deliverable that runs locally
without a GPU once the v3 predictions CSV exists. The two training
ablations (no_vsn, global_norm) require a GPU (Vertex AI Custom Job).

USAGE
    # default: read results/tft_v3_predictions_2025.csv, sweep
    # confidence thresholds and write the sensitivity table
    python -m training.tft_ablations.no_gating_eval

    # explicit input
    python -m training.tft_ablations.no_gating_eval \\
        --predictions results/tft_v3_predictions_2025.csv \\
        --out results/tft_ablations/confidence_gating_sweep.csv

OUTPUTS
    results/tft_ablations/confidence_gating_sweep.csv
    results/tft_ablations/confidence_gating_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# Mirror agents/technical_agent.py:449 confidence formula:
#   confidence = clip(1 - (Q90 - Q10) / 0.10, 0.05, 0.99)
# A wider quantile interval => lower confidence. The denominator 0.10
# is the "expected" Q10-Q90 spread for a typical 7-day BVMT return
# (volatility ~3-5%, x2 for 80% interval -> ~10%).
CONFIDENCE_DENOM = 0.10
CONFIDENCE_LO = 0.05
CONFIDENCE_HI = 0.99


def derive_confidence(q10: np.ndarray, q90: np.ndarray) -> np.ndarray:
    width = np.maximum(q90 - q10, 0.0)
    raw = 1.0 - width / CONFIDENCE_DENOM
    return np.clip(raw, CONFIDENCE_LO, CONFIDENCE_HI)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--predictions", type=Path,
        default=ROOT / "results" / "tft_v3_predictions_2025.csv",
        help=("Per-row TFT v3 predictions CSV. Schema must include "
              "columns: q10, q50, q90, true_direction (0/1) "
              "or true_return_7d (float)."))
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "results" / "tft_ablations")
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.00, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
                   help="Confidence thresholds tau to sweep.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.predictions.exists():
        print(f"\nERROR: predictions file not found: {args.predictions}")
        print("Run training/evaluate_tft_v3_per_row.py on Vertex AI first")
        print("(needs the v3 checkpoint at models/tft_bvmt_bestv3.ckpt).")
        print("Falling back to skeleton output so downstream scripts can")
        print("be wired up while waiting for the cluster run.")
        skeleton = pd.DataFrame({
            "tau": args.thresholds,
            "coverage": [None] * len(args.thresholds),
            "accuracy": [None] * len(args.thresholds),
            "n_kept": [0] * len(args.thresholds),
            "n_total": [0] * len(args.thresholds),
        })
        out_csv = args.out_dir / "confidence_gating_sweep.csv"
        skeleton.to_csv(out_csv, index=False)
        with open(args.out_dir / "confidence_gating_summary.json", "w") as f:
            json.dump({
                "status": "pending_predictions",
                "predictions_path": str(args.predictions),
                "thresholds": args.thresholds,
                "note": ("This script will populate the table once "
                         "results/tft_v3_predictions_2025.csv is generated "
                         "by training/evaluate_tft_v3_per_row.py."),
            }, f, indent=2)
        print(f"  Wrote skeleton: {out_csv}")
        return 0

    df = pd.read_csv(args.predictions)
    required = {"q10", "q50", "q90"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: predictions CSV missing columns: {missing}")
        return 1

    if "true_direction" in df.columns:
        y_true = df["true_direction"].astype(int).to_numpy()
    elif "true_return_7d" in df.columns:
        y_true = (df["true_return_7d"].astype(float).to_numpy() > 0).astype(int)
    else:
        print("ERROR: predictions CSV must have true_direction or true_return_7d")
        return 1

    q10 = df["q10"].astype(float).to_numpy()
    q50 = df["q50"].astype(float).to_numpy()
    q90 = df["q90"].astype(float).to_numpy()
    pred_dir = (q50 >= 0).astype(int)
    confidence = derive_confidence(q10, q90)

    n_total = len(df)
    rows = []
    for tau in args.thresholds:
        mask = confidence >= tau
        n_kept = int(mask.sum())
        if n_kept == 0:
            rows.append({"tau": tau, "coverage": 0.0,
                         "accuracy": None, "n_kept": 0, "n_total": n_total})
            continue
        acc = float((pred_dir[mask] == y_true[mask]).mean())
        rows.append({
            "tau": tau,
            "coverage": round(n_kept / n_total, 4),
            "accuracy": round(acc, 4),
            "n_kept": n_kept,
            "n_total": n_total,
        })

    out_csv = args.out_dir / "confidence_gating_sweep.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    summary = {
        "status": "ok",
        "predictions_path": str(args.predictions),
        "n_total": n_total,
        "confidence_stats": {
            "mean": round(float(confidence.mean()), 4),
            "p10": round(float(np.quantile(confidence, 0.10)), 4),
            "p50": round(float(np.quantile(confidence, 0.50)), 4),
            "p90": round(float(np.quantile(confidence, 0.90)), 4),
        },
        "sweep": rows,
    }
    with open(args.out_dir / "confidence_gating_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Confidence gating sensitivity ===")
    print(f"  n_total = {n_total:,}")
    print(f"  tau    coverage   accuracy   (n_kept)")
    for r in rows:
        cov = "n/a" if r["coverage"] is None else f"{r['coverage']*100:5.1f}%"
        acc = "n/a" if r["accuracy"] is None else f"{r['accuracy']*100:5.2f}%"
        print(f"  {r['tau']:.2f}  {cov:>7}   {acc:>7}   ({r['n_kept']:,})")
    print(f"\n  Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
