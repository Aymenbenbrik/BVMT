"""P1.1 + P1.2: empirical 80% coverage + per-stock TFT v3 IQR.

P1.1 — Calibration:
  Read baseline_test_predictions.csv (q10, q50, q90, true_return_7d).
  Compute fraction of rows where q10 <= true_return_7d <= q90 — target 80%.

P1.2 — Per-stock IQR:
  Read vertex_slice_harmonized_predictions.csv (pred_tft_v3, true_direction).
  Per-ticker accuracy, then q25/median/q75 across tickers.
"""
from __future__ import annotations
import sys
from pathlib import Path
import csv
from collections import defaultdict
from statistics import median, quantiles

ROOT = Path(r"C:/Users/aymen/OneDrive/Bureau/recherche/IEEE/bvmt-project-main/bvmt-project-main")
PRED_QUANTILES = ROOT / "results" / "tft_ablations" / "vertex" / "baseline_test_predictions.csv"
PRED_HARMONIZED = ROOT / "results" / "baselines" / "vertex_slice_harmonized_predictions.csv"


def p1_1_calibration() -> None:
    inside = 0
    total = 0
    inside_dedup = 0
    total_dedup = 0
    seen: set[tuple[str, str]] = set()

    skipped = 0
    with PRED_QUANTILES.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                q10 = float(row["q10"])
                q90 = float(row["q90"])
                r = float(row["true_return_7d"])
            except ValueError:
                skipped += 1
                continue
            total += 1
            if q10 <= r <= q90:
                inside += 1
            key = (row["ticker"], row["date"])
            if key not in seen:
                seen.add(key)
                total_dedup += 1
                if q10 <= r <= q90:
                    inside_dedup += 1
    print(f"  (Skipped {skipped} rows with non-parseable values)")

    cov_all = 100 * inside / total
    cov_dedup = 100 * inside_dedup / total_dedup
    print(f"P1.1 — Empirical 80% coverage")
    print(f"  All rows (with multi-horizon duplicates):  N={total}, inside [q10,q90]={inside}, coverage={cov_all:.2f}%")
    print(f"  Deduplicated (ticker,date):                N={total_dedup}, inside={inside_dedup}, coverage={cov_dedup:.2f}%")
    print(f"  Target: 80%; gap: {cov_dedup - 80:+.2f} pp (dedup), {cov_all - 80:+.2f} pp (raw)")


def p1_2_per_stock() -> None:
    per_ticker: dict[str, list[int]] = defaultdict(list)

    with PRED_HARMONIZED.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row["ticker"]
            try:
                truth = int(row["true_direction"])
                pred = int(row["pred_tft_v3"])
            except (ValueError, KeyError):
                continue
            per_ticker[ticker].append(1 if pred == truth else 0)

    per_ticker_acc: dict[str, tuple[float, int]] = {
        t: (100 * sum(v) / len(v), len(v))
        for t, v in per_ticker.items()
        if len(v) >= 1
    }

    acc_list = [a for (a, _n) in per_ticker_acc.values()]
    acc_list.sort()
    n_tickers = len(acc_list)
    med = median(acc_list)
    if n_tickers >= 4:
        q25, _q50, q75 = quantiles(acc_list, n=4, method="inclusive")
    else:
        q25 = q75 = med
    iqr = q75 - q25

    above_58 = sum(1 for a in acc_list if a > 58.1)
    above_5279 = sum(1 for a in acc_list if a > 52.79)
    above_50 = sum(1 for a in acc_list if a > 50.0)

    print(f"\nP1.2 — Per-stock TFT v3 accuracy on harmonized slice")
    print(f"  Tickers evaluated: {n_tickers}")
    print(f"  Min / Max:         {min(acc_list):.2f}% / {max(acc_list):.2f}%")
    print(f"  Q25 / Median / Q75: {q25:.2f}% / {med:.2f}% / {q75:.2f}%")
    print(f"  IQR:               {iqr:.2f} pp")
    print(f"  Fraction > 50%:    {100*above_50/n_tickers:.1f}% ({above_50}/{n_tickers})")
    print(f"  Fraction > 52.79% (LogReg pooled): {100*above_5279/n_tickers:.1f}% ({above_5279}/{n_tickers})")
    print(f"  Fraction > 58.1% (always-DOWN historical floor): {100*above_58/n_tickers:.1f}% ({above_58}/{n_tickers})")

    n_lt500 = sum(1 for (_a, n) in per_ticker_acc.values() if n < 500)
    print(f"\n  Tickers with <500 rows on this slice: {n_lt500}/{n_tickers}")

    print(f"\n  Per-ticker accuracies (sorted):")
    for t, (a, n) in sorted(per_ticker_acc.items(), key=lambda x: x[1][0]):
        print(f"    {t:30s}  {a:6.2f}%  (N={n})")


if __name__ == "__main__":
    p1_1_calibration()
    p1_2_per_stock()
