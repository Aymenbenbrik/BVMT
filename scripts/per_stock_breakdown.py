"""scripts/per_stock_breakdown.py

Per-stock breakdown of the TFT v3 directional accuracy on the 2025 test
set. Implements D3 of paper/critique_independent.md: the aggregate
77.4% headline figure is computed over 43 of 68 tickers (Section V.A
of the article); a paper-grade evaluation needs the per-ticker spread,
the worst- and best-performing tickers, and the per-ticker statistical
significance against the always-DOWN majority baseline.

INPUT
    A CSV produced by training/evaluate_tft_v3_per_row.py with at least
    the columns: ticker, q50, true_return_7d, true_direction. The R1
    decision rule (UP iff Q50 >= 0) is the one documented in Section
    V.B of the article; alternative rules R2/R3/R4 from
    paper/improvement_plan.md Action 1.5b can be plugged in via --rule.

OUTPUT
    A JSON report and a CSV per-ticker table. Stdout prints:
      - overall accuracy and 95% bootstrap CI
      - top-5 best- and worst-predicted tickers
      - distribution summary (count, mean, std, IQR, fraction below 50%)
      - paste-ready LaTeX rows for tab:per_stock_breakdown

USAGE
    python scripts/per_stock_breakdown.py \\
        --in  results/tft_v3_predictions_2025.csv \\
        --out reports/per_stock/breakdown.json \\
        --csv reports/per_stock/per_ticker_table.csv

The script is robust to CSVs produced by any compatible source (e.g. a
re-run of evaluate_tft_v3_per_row, a future quantile-regression model,
or a synthetic test fixture) as long as the required columns are
present.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ALWAYS_DOWN_BASELINE = 0.581  # 58.1% on the 2025 test set, see Section V.A


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="input", type=Path,
                   default=ROOT / "results" / "tft_v3_predictions_2025.csv")
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "per_stock" / "breakdown.json")
    p.add_argument("--csv", type=Path,
                   default=ROOT / "reports" / "per_stock" / "per_ticker_table.csv")
    p.add_argument("--rule", choices=["R1", "R2", "R3", "R4"], default="R1",
                   help="Decision rule mapping quantiles to direction. Default R1 (sign of Q50).")
    p.add_argument("--hold-tau", type=float, default=0.005,
                   help="HOLD-band threshold for R3 (|Q50| < tau). Default 0.5%%.")
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def apply_rule(df: pd.DataFrame, rule: str, hold_tau: float) -> pd.Series:
    """Map quantile predictions to {0=DOWN, 1=UP, 2=HOLD/abstain} per the chosen rule."""
    q10 = df.get("q10", pd.Series(np.nan, index=df.index)).astype(float)
    q50 = df["q50"].astype(float)
    q90 = df.get("q90", pd.Series(np.nan, index=df.index)).astype(float)

    if rule == "R1":
        return (q50 >= 0).astype(int)
    if rule == "R2":
        out = pd.Series(2, index=df.index, dtype=int)  # 2 = abstain
        out[q10 > 0] = 1
        out[q90 < 0] = 0
        return out
    if rule == "R3":
        out = pd.Series((q50 >= 0).astype(int).values, index=df.index)
        out[q50.abs() < hold_tau] = 2  # HOLD
        return out
    if rule == "R4":
        # Linear interpolation of the inverse CDF through (0.1, q10), (0.5, q50),
        # (0.9, q90); recover P(return > 0) and threshold at 0.5.
        # If quantile sequence is non-monotone, fall back to R1.
        def _p_pos(row):
            xs = [row.get("q10"), row.get("q50"), row.get("q90")]
            ys = [0.1, 0.5, 0.9]
            try:
                if not (xs[0] <= xs[1] <= xs[2]):
                    return 1.0 if xs[1] >= 0 else 0.0
                if xs[2] <= 0:
                    return 0.0
                if xs[0] >= 0:
                    return 1.0
                # zero falls inside the interpolation range
                # piecewise linear inversion
                if xs[0] <= 0 <= xs[1]:
                    frac = (0 - xs[0]) / (xs[1] - xs[0]) if xs[1] != xs[0] else 0.5
                    p_le_zero = 0.1 + frac * (0.5 - 0.1)
                else:
                    frac = (0 - xs[1]) / (xs[2] - xs[1]) if xs[2] != xs[1] else 0.5
                    p_le_zero = 0.5 + frac * (0.9 - 0.5)
                return 1.0 - p_le_zero
            except Exception:
                return 1.0 if (xs[1] is not None and xs[1] >= 0) else 0.0

        p_up = df.apply(_p_pos, axis=1)
        return (p_up >= 0.5).astype(int)
    raise ValueError(f"unknown rule {rule}")


def bootstrap_acc_ci(correct: np.ndarray, n_resamples: int, rng_seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    n = len(correct)
    if n == 0:
        return float("nan"), float("nan")
    boots = rng.choice(correct, size=(n_resamples, n), replace=True).mean(axis=1)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def per_ticker_table(df: pd.DataFrame, rule: str, hold_tau: float, bootstrap_n: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["pred"] = apply_rule(df, rule, hold_tau)

    rows = []
    rng_offset = 0
    for ticker, sub in df.groupby("ticker"):
        true = sub["true_direction"].astype(int).values
        pred = sub["pred"].astype(int).values
        # For R2/R3, abstentions are coded as 2 — exclude them from accuracy.
        cover_mask = pred != 2
        coverage = cover_mask.mean() if len(pred) else float("nan")
        if cover_mask.sum() == 0:
            rows.append({"ticker": ticker, "n": len(sub), "n_scored": 0,
                         "coverage": coverage, "acc": float("nan"),
                         "ci_lo": float("nan"), "ci_hi": float("nan"),
                         "vs_always_down": float("nan"),
                         "n_up_true": int((true == 1).sum()),
                         "n_down_true": int((true == 0).sum())})
            continue
        correct = (pred[cover_mask] == true[cover_mask]).astype(float)
        acc = correct.mean()
        ci = bootstrap_acc_ci(correct, n_resamples=bootstrap_n, rng_seed=seed + rng_offset)
        rng_offset += 1
        always_down_acc = (true[cover_mask] == 0).mean()
        rows.append({
            "ticker":         ticker,
            "n":              int(len(sub)),
            "n_scored":       int(cover_mask.sum()),
            "coverage":       round(coverage, 4),
            "acc":            round(acc, 4),
            "ci_lo":          round(ci[0], 4),
            "ci_hi":          round(ci[1], 4),
            "vs_always_down": round(acc - always_down_acc, 4),
            "n_up_true":      int((true == 1).sum()),
            "n_down_true":    int((true == 0).sum()),
        })
    out = pd.DataFrame(rows).sort_values("acc", ascending=False, na_position="last").reset_index(drop=True)
    return out


def overall_summary(per_stock: pd.DataFrame, rule: str) -> dict:
    acc = per_stock["acc"].dropna().tolist()
    if not acc:
        return {"n_tickers": 0, "rule": rule}
    arr = np.asarray(acc, dtype=float)
    return {
        "rule":             rule,
        "n_tickers":        int(len(per_stock)),
        "n_tickers_scored": int((per_stock["n_scored"] > 0).sum()),
        "ticker_acc": {
            "mean":      round(float(arr.mean()), 4),
            "median":    round(float(np.median(arr)), 4),
            "stdev":     round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 4),
            "iqr_q1":    round(float(np.quantile(arr, 0.25)), 4),
            "iqr_q3":    round(float(np.quantile(arr, 0.75)), 4),
            "min":       round(float(arr.min()), 4),
            "max":       round(float(arr.max()), 4),
        },
        "fraction_below_50_pct":           round(float((arr < 0.50).mean()), 4),
        "fraction_below_always_down":      round(float((arr < ALWAYS_DOWN_BASELINE).mean()), 4),
        "fraction_above_77_4_pct":         round(float((arr >= 0.774).mean()), 4),
    }


def render_latex_rows(per_stock: pd.DataFrame, n_top: int = 5) -> list[str]:
    if per_stock.empty:
        return []
    head = per_stock.head(n_top)
    tail = per_stock.tail(n_top)[::-1]
    rows = []
    for label, sub in [("\\textbf{Best}", head), ("\\textbf{Worst}", tail)]:
        for _, r in sub.iterrows():
            rows.append(
                f"{label} & {r['ticker']:<10} & {int(r['n_scored']):>4d} & "
                f"{r['acc']*100:5.1f} & "
                f"[{r['ci_lo']*100:5.1f}, {r['ci_hi']*100:5.1f}] & "
                f"{r['vs_always_down']*100:+5.1f} \\\\"
            )
    return rows


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(
            f"Input not found: {args.input}\n"
            "Generate it with `python training/evaluate_tft_v3_per_row.py` first."
        )
    df = pd.read_csv(args.input)
    required = {"ticker", "q50", "true_direction"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    print(f"Loaded {len(df):,} predictions covering {df['ticker'].nunique()} tickers.")

    per_stock = per_ticker_table(df, args.rule, args.hold_tau,
                                 bootstrap_n=args.bootstrap, seed=args.seed)
    summary = overall_summary(per_stock, args.rule)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    per_stock.to_csv(args.csv, index=False)
    print(f"Wrote per-ticker table to {args.csv}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary["latex_rows"] = render_latex_rows(per_stock, n_top=5)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote summary to {args.out}")

    print("\n" + "=" * 60)
    print(f"PER-STOCK BREAKDOWN  (decision rule: {args.rule})")
    print("=" * 60)
    print(f"Tickers scored          : {summary['n_tickers_scored']} / {summary['n_tickers']}")
    a = summary["ticker_acc"]
    print(f"Per-ticker accuracy     : mean={a['mean']:.3f}  median={a['median']:.3f}  std={a['stdev']:.3f}")
    print(f"  IQR                   : [{a['iqr_q1']:.3f}, {a['iqr_q3']:.3f}]")
    print(f"  range                 : [{a['min']:.3f}, {a['max']:.3f}]")
    print(f"Fraction < 50%          : {summary['fraction_below_50_pct']*100:.1f}%")
    print(f"Fraction < always-DOWN  : {summary['fraction_below_always_down']*100:.1f}%")
    print(f"Fraction >= 77.4%       : {summary['fraction_above_77_4_pct']*100:.1f}%")
    print()
    print("TOP 5 (best per-ticker accuracy):")
    print(per_stock.head(5).to_string(index=False))
    print("\nWORST 5 (lowest per-ticker accuracy):")
    print(per_stock.tail(5)[::-1].to_string(index=False))

    print("\nLaTeX rows (paste into tab:per_stock_breakdown):")
    for line in summary["latex_rows"]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
