"""training/baselines/trivial.py

Action 1.1 of paper/improvement_plan.md: four no-ML baselines on the
same walk-forward test split as the TFT v3 model.

Baselines (all read from data/features/tft_features.csv columns that
the cleaning pipeline already exposes):

    always_up       : predict UP for every row.
    always_down     : predict DOWN for every row. Given the BVMT 41.9%
                      UP imbalance, this is the relevant naive baseline
                      (cf. Section V.A of the article and tab:baselines).
    momentum_5_20   : predict UP iff ma_5 > ma_20.
    ma20_crossover  : predict UP iff close_price > ma_20.

The output is two artefacts:
    results/baselines/trivial_predictions.csv
        per-row predictions for the four baselines, ready for the
        per-stock and McNemar pipelines (Action 1.5b, 1.8).

    results/baselines/trivial_summary.json
        one block per baseline with the metrics defined in Action 1.5
        (accuracy, 95% bootstrap CI, per-class precision/recall/F1,
        confusion matrix, per-stock spread); a pairwise McNemar table
        comparing every pair; a paste-ready LaTeX table for the
        article's tab:baselines extension.

USAGE
    python -m training.baselines.trivial \\
        --data data/features/tft_features.csv \\
        --out  results/baselines/

NOTES
    The script does not split-leak: it filters rows where date >=
    VAL_END (== 2025-01-01) and uses only columns derived causally
    (Section III.B of the article). The same VAL_END constant matches
    train_tft_v3.py.
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

from training.baselines.metrics import (  # noqa: E402
    evaluate_model,
    pairwise_mcnemar_table,
)

VAL_END = "2025-01-01"  # matches train_tft_v3.py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "features" / "tft_features.csv")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "baselines")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_test_split(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    test = df[df["date"] >= pd.Timestamp(VAL_END)].copy()
    if test.empty:
        raise SystemExit(
            f"Test split is empty. Check that {data_path} contains rows with date >= {VAL_END}."
        )
    test["direction_7d"] = test["direction_7d"].astype(int)
    return test.reset_index(drop=True)


def build_baselines(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return per-row predictions for the four trivial baselines."""
    n = len(df)
    return {
        "always_up":      np.ones(n, dtype=int),
        "always_down":    np.zeros(n, dtype=int),
        "momentum_5_20":  (df["ma_5"].values > df["ma_20"].values).astype(int),
        "ma20_crossover": (df["close_price"].values > df["ma_20"].values).astype(int),
    }


def render_latex_rows(summaries: list[dict], baseline_label: str = "Always-DOWN") -> list[str]:
    """Paste-ready LaTeX rows for the tab:baselines extension. Each row
    reports model, n, accuracy %, 95% CI, F1_UP, F1_DOWN, macro-F1, lift
    over Always-DOWN."""
    if not summaries:
        return []
    ad = next((s for s in summaries if s["name"] == "always_down"), None)
    ad_acc = ad["accuracy"] if ad else 0.581
    rows = []
    for s in summaries:
        acc_pct = s["accuracy"] * 100
        ci_lo = s["ci95_lo"] * 100
        ci_hi = s["ci95_hi"] * 100
        f1_up = s["per_class"]["UP"]["f1"]
        f1_dn = s["per_class"]["DOWN"]["f1"]
        macro = s["per_class"]["macro_f1"]
        lift = (s["accuracy"] - ad_acc) * 100
        rows.append(
            f"{s['name'].replace('_', ' '):<20} & {s['n']:>5d} & "
            f"{acc_pct:5.2f} & [{ci_lo:5.2f}, {ci_hi:5.2f}] & "
            f"{f1_up:.3f} & {f1_dn:.3f} & {macro:.3f} & "
            f"{lift:+5.2f} \\\\"
        )
    return rows


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading test split from {args.data} ...")
    df = load_test_split(args.data)
    print(f"  Test rows : {len(df):,}")
    print(f"  Tickers   : {df['ticker'].nunique()}")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    y_true = df["direction_7d"].values
    tickers = df["ticker"].values

    preds = build_baselines(df)

    # 1) Persist per-row predictions for downstream pipelines
    pred_csv = args.out / "trivial_predictions.csv"
    pred_df = df[["ticker", "ticker_id", "date", "time_idx", "direction_7d"]].copy()
    pred_df["date"] = pred_df["date"].dt.strftime("%Y-%m-%d")
    pred_df = pred_df.rename(columns={"direction_7d": "true_direction"})
    for name, p in preds.items():
        pred_df[f"pred_{name}"] = p
    pred_df.to_csv(pred_csv, index=False)
    print(f"\nWrote per-row predictions: {pred_csv}")

    # 2) Per-model metrics
    summaries = []
    for name, p in preds.items():
        s = evaluate_model(name, y_true, p, tickers=tickers,
                           bootstrap_n=args.bootstrap, rng_seed=args.seed)
        summaries.append(s)

    # 3) Pairwise McNemar
    mcnemar_pairs = pairwise_mcnemar_table(y_true, preds)

    # 4) Console summary
    print("\n" + "=" * 78)
    print(f"{'Baseline':<20} {'n':>6} {'acc':>7} {'ci_lo':>7} {'ci_hi':>7} {'F1_UP':>6} {'F1_DN':>6} {'macroF1':>7}")
    print("-" * 78)
    for s in summaries:
        pc = s["per_class"]
        print(
            f"{s['name']:<20} {s['n']:>6d} "
            f"{s['accuracy']*100:6.2f}% {s['ci95_lo']*100:6.2f}% {s['ci95_hi']*100:6.2f}% "
            f"{pc['UP']['f1']:>6.3f} {pc['DOWN']['f1']:>6.3f} {pc['macro_f1']:>7.3f}"
        )
    print("=" * 78)

    # 5) Pairwise McNemar
    print("\nPairwise McNemar (two-sided):")
    print(f"{'A':<18} vs {'B':<18}  {'a-r,b-w':>9} {'a-w,b-r':>9} {'stat':>9} {'p-value':>10}  test")
    for r in mcnemar_pairs:
        print(
            f"{r['a']:<18} vs {r['b']:<18}  {r['b_wrong_a_right']:>9d} {r['a_wrong_b_right']:>9d} "
            f"{r['stat']:>9.4f} {r['p_value']:>10.6f}  {r['test']}"
        )

    # 6) Per-stock spread per baseline
    print("\nPer-stock spread (per baseline):")
    print(f"{'Baseline':<20} {'n_tickers':>10} {'mean':>7} {'median':>8} {'IQR':>10} {'<50%':>6} {'<58.1%':>8}")
    for s in summaries:
        sp = s.get("per_stock", {}).get("spread", {})
        if not sp:
            continue
        iqr = f"[{sp['q1']*100:.1f},{sp['q3']*100:.1f}]"
        print(
            f"{s['name']:<20} {sp['n_tickers']:>10d} "
            f"{sp['mean']*100:6.2f}% {sp['median']*100:7.2f}% {iqr:>10} "
            f"{sp['frac_below_50']*100:5.1f}% {sp['frac_below_581']*100:7.1f}%"
        )

    # 7) Save JSON + LaTeX
    summary_obj = {
        "test_set": {
            "n_rows": int(len(df)),
            "n_tickers": int(df["ticker"].nunique()),
            "date_min": df["date"].min().strftime("%Y-%m-%d"),
            "date_max": df["date"].max().strftime("%Y-%m-%d"),
            "up_ratio": round(float((y_true == 1).mean()), 4),
        },
        "baselines": summaries,
        "pairwise_mcnemar": mcnemar_pairs,
        "latex_rows_baseline_table": render_latex_rows(summaries),
    }
    out_json = args.out / "trivial_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2)
    print(f"\nWrote summary JSON: {out_json}")

    print("\nLaTeX rows (paste into tab:baselines_phase1):")
    for line in summary_obj["latex_rows_baseline_table"]:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
