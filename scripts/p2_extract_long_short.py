"""P2.2: Extract long-short NET backtest values from economic_summary.json.

Produces a sorted table for the paper showing how the ranking changes
between long-only and long-short regimes.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(r"C:/Users/aymen/OneDrive/Bureau/recherche/IEEE/bvmt-project-main/bvmt-project-main")
SUMMARY = ROOT / "results" / "baselines" / "economic_summary.json"

with SUMMARY.open("r", encoding="utf-8") as f:
    data = json.load(f)

per_model = data["per_model"]

def pretty(name: str) -> str:
    name = name.replace("pred_", "")
    return name.replace("_", " ")

rows_ls = []
rows_lo = []
for model, strats in per_model.items():
    ls = strats.get("long_short", {}).get("net", {})
    lo = strats.get("long_only", {}).get("net", {})
    if ls:
        rows_ls.append({
            "model": pretty(model),
            "cum": ls["cumulative_return"],
            "sharpe": ls["sharpe_ratio"],
            "mdd": ls["max_drawdown"],
            "dsr_p": ls["deflated_sharpe"]["p_value"],
        })
    if lo:
        rows_lo.append({
            "model": pretty(model),
            "cum": lo["cumulative_return"],
            "sharpe": lo["sharpe_ratio"],
            "mdd": lo["max_drawdown"],
            "dsr_p": lo["deflated_sharpe"]["p_value"],
        })

rows_ls.sort(key=lambda r: r["sharpe"], reverse=True)
rows_lo.sort(key=lambda r: r["sharpe"], reverse=True)

print("=== LONG-SHORT NET (sorted by Sharpe) ===")
print(f"{'Model':<22} {'Cum%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'DSR p':>8}")
print("-" * 60)
for r in rows_ls:
    print(f"{r['model']:<22} {r['cum']*100:>7.2f}% {r['sharpe']:>7.2f} {r['mdd']*100:>7.2f}% {r['dsr_p']:>8.4f}")

print("\n=== LONG-ONLY NET (sorted by Sharpe, paper's Table IX) ===")
print(f"{'Model':<22} {'Cum%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'DSR p':>8}")
print("-" * 60)
for r in rows_lo:
    print(f"{r['model']:<22} {r['cum']*100:>7.2f}% {r['sharpe']:>7.2f} {r['mdd']*100:>7.2f}% {r['dsr_p']:>8.4f}")

# Rank correlation between the two regimes
lo_rank = {r["model"]: i for i, r in enumerate(rows_lo)}
ls_rank = {r["model"]: i for i, r in enumerate(rows_ls)}
common = set(lo_rank) & set(ls_rank)
spearman_num = 0.0
n = len(common)
ranks_lo = sorted(lo_rank.items(), key=lambda x: x[1])
ranks_ls = sorted(ls_rank.items(), key=lambda x: x[1])
# Compute Spearman manually
diffs = [(lo_rank[m] - ls_rank[m])**2 for m in common]
rho = 1 - 6 * sum(diffs) / (n * (n**2 - 1))
print(f"\nSpearman rank correlation between LO and LS Sharpe orderings (n={n}): rho = {rho:.4f}")

# Count clearance of DSR p < 0.05 in each regime
n_ls_clear = sum(1 for r in rows_ls if r["dsr_p"] < 0.05)
n_lo_clear = sum(1 for r in rows_lo if r["dsr_p"] < 0.05)
print(f"\nDSR p < 0.05 clearance:")
print(f"  long-short NET: {n_ls_clear}/{len(rows_ls)}")
print(f"  long-only  NET: {n_lo_clear}/{len(rows_lo)}")
