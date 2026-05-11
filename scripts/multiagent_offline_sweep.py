"""scripts/multiagent_offline_sweep.py

Offline subset of the multi-agent ablation sweep (A1-A9) described in
Section V.G of the article. Realises the three configurations that depend
only on TFT v3 per-row predictions and raw price history:

    A1  -- TFT only (reference, Vertex-reproducible)
    A4  -- TFT + Graph (proxy: 60-day Pearson + TFT-weighted neighbors)
    A8  -- A4 with confidence gating disabled (graph signal always used)

Also reports a graph-only classifier for context, and a sensitivity sweep
over (w_graph, threshold) to quantify the maximum effect the graph signal
can have on the binary technical decision.

The remaining six configurations (A2, A3, A5, A6, A7, A9) require the
deployed PostgreSQL database (`news_articles` pre-scored sentiment +
`financial_ratios`) and the trained XGBoost fundamental checkpoint, none
of which is shipped as offline data; they remain DB-dependent.

INPUT
    data/features/tft_features.csv
        - 105k rows of per-(ticker, date) features including `daily_return`
    results/baselines/vertex_slice_harmonized_predictions.csv
        - 5,582 row-level TFT v3 predictions on the harmonized 2025 slice

OUTPUT
    results/baselines/multiagent_offline_sweep_predictions.csv
        - per-row CSV with TFT v3 prediction + graph score/confidence + A4/A8
    results/baselines/multiagent_offline_sweep.json
        - summary metrics, weight-sensitivity table, DB-dependency note

USAGE
    python scripts/multiagent_offline_sweep.py

Author: Aymen Ben Brik <aymen.benbrik@esprit.tn>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2dist
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "features" / "tft_features.csv"
SLICE_PATH = ROOT / "results" / "baselines" / "vertex_slice_harmonized_predictions.csv"
PRED_OUT = ROOT / "results" / "baselines" / "multiagent_offline_sweep_predictions.csv"
SUMMARY_OUT = ROOT / "results" / "baselines" / "multiagent_offline_sweep.json"

WINDOW = 60
TOP_K = 10
MIN_CORR = 0.30

# Orchestrator gate and weights (from agents/orchestrator.py)
GATE_MIN_CONF = 0.20
GATE_MIN_ABS_SCORE = 0.08
W_TECH_DEPLOYED = 0.34
W_GRAPH_DEPLOYED = 0.20  # re-normalised to 0.34/0.54 vs 0.20/0.54 in 2-agent mode
DECISION_THRESH = 0.50  # binary cut


def compute_graph_signal(slice_df: pd.DataFrame, returns: pd.DataFrame,
                         tft_lookup: dict) -> pd.DataFrame:
    """For each (ticker, date), compute a graph_score + confidence by
    aggregating top-K correlated neighbors' TFT v3 R1 predictions."""
    out = []
    n = len(slice_df)
    for i, row in slice_df.iterrows():
        if i % 500 == 0:
            print(f"  row {i}/{n}", flush=True)
        ticker = row["ticker"]
        date = row["date"]
        end = date - pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=120)  # generous calendar buffer
        window = returns.loc[start:end].tail(WINDOW)
        if len(window) < 30 or ticker not in window.columns:
            out.append((0.0, 0.0, 0))
            continue
        valid = window.dropna(axis=1, thresh=int(0.7 * WINDOW))
        if ticker not in valid.columns:
            out.append((0.0, 0.0, 0))
            continue
        target = valid[ticker]
        corrs = valid.corrwith(target).drop(ticker, errors="ignore").dropna()
        if corrs.empty:
            out.append((0.0, 0.0, 0))
            continue
        top = corrs.reindex(corrs.abs().sort_values(ascending=False).index).head(TOP_K)
        top = top[top.abs() >= MIN_CORR]
        if top.empty:
            out.append((0.0, 0.0, 0))
            continue
        preds, weights = [], []
        for n_ticker, corr in top.items():
            key = (n_ticker, date)
            if key in tft_lookup:
                preds.append(2 * tft_lookup[key] - 1)
                weights.append(corr)
        if not preds:
            out.append((0.0, 0.0, 0))
            continue
        preds_arr = np.asarray(preds, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)
        graph_score = float(np.sum(weights_arr * preds_arr) / np.sum(np.abs(weights_arr)))
        confidence = float(np.mean(np.abs(weights_arr)))
        pred = 1 if graph_score >= 0 else 0
        out.append((graph_score, confidence, pred))
    return pd.DataFrame(out, columns=["graph_score", "graph_conf", "pred_graph_only"])


def combine_orchestrator(tft_pred: int, graph_score: float, graph_conf: float,
                         w_tech: float, w_graph: float,
                         gate_active: bool = True,
                         decision_thresh: float = 0.50) -> int:
    """Replicate the binary-decision logic of OrchestratorAgent._assemble_final_result
    restricted to the 2-agent (technical + graph) case."""
    tech_score = float(tft_pred)
    if gate_active:
        if graph_conf < GATE_MIN_CONF or abs(graph_score) < GATE_MIN_ABS_SCORE:
            return int(tech_score >= decision_thresh)
    g = (graph_score + 1.0) / 2.0  # to [0, 1]
    total = w_tech + w_graph
    combined = (w_tech * tech_score + w_graph * g) / total
    return int(combined >= decision_thresh)


def boot_ci(y: pd.Series, p: pd.Series, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_arr = y.values
    p_arr = p.values
    n = len(y_arr)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        accs[i] = (y_arr[idx] == p_arr[idx]).mean()
    return float((y_arr == p_arr).mean() * 100), float(np.percentile(accs, 2.5) * 100), float(np.percentile(accs, 97.5) * 100)


def mcnemar(y: pd.Series, a: pd.Series, b: pd.Series):
    a_right = (a == y)
    b_right = (b == y)
    n01 = int(((~a_right) & b_right).sum())
    n10 = int((a_right & (~b_right)).sum())
    n = n01 + n10
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / n
    return float(chi2), float(1 - chi2dist.cdf(chi2, df=1))


def macro_f1(y: pd.Series, p: pd.Series) -> float:
    return float((f1_score(y, p, pos_label=1) + f1_score(y, p, pos_label=0)) / 2)


def main() -> int:
    t0 = time.time()

    print("Loading features...")
    feat = pd.read_csv(FEATURES_PATH, usecols=["ticker", "date", "daily_return"])
    feat["date"] = pd.to_datetime(feat["date"])
    returns = feat.pivot_table(index="date", columns="ticker", values="daily_return", aggfunc="first").sort_index()
    print(f"  returns matrix: {returns.shape[0]} dates x {returns.shape[1]} tickers")

    print("Loading harmonized test slice...")
    slice_df = pd.read_csv(SLICE_PATH)
    slice_df["date"] = pd.to_datetime(slice_df["date"])
    print(f"  test rows: {len(slice_df)}")

    tft_lookup = {(r["ticker"], r["date"]): int(r["pred_tft_v3"]) for _, r in slice_df.iterrows()}

    print("\nComputing graph signal (60-day Pearson, top-10 neighbors, |rho| >= 0.30)...")
    graph_df = compute_graph_signal(slice_df, returns, tft_lookup)
    slice_df = pd.concat([slice_df.reset_index(drop=True), graph_df], axis=1)

    covered = int((slice_df["graph_conf"] > 0).sum())
    passing_gate = int(((slice_df["graph_conf"] >= GATE_MIN_CONF) & (slice_df["graph_score"].abs() >= GATE_MIN_ABS_SCORE)).sum())
    print(f"  coverage : {covered}/{len(slice_df)} ({100 * covered / len(slice_df):.1f}%)")
    print(f"  passing orchestrator gate: {passing_gate} ({100 * passing_gate / len(slice_df):.1f}%)")

    print("\nBuilding A1, A4, A8 predictions...")
    slice_df["pred_A1"] = slice_df["pred_tft_v3"]
    slice_df["pred_A4"] = [
        combine_orchestrator(r["pred_tft_v3"], r["graph_score"], r["graph_conf"],
                             W_TECH_DEPLOYED, W_GRAPH_DEPLOYED, gate_active=True)
        for _, r in slice_df.iterrows()
    ]
    slice_df["pred_A8"] = [
        combine_orchestrator(r["pred_tft_v3"], r["graph_score"], r["graph_conf"],
                             W_TECH_DEPLOYED, W_GRAPH_DEPLOYED, gate_active=False)
        for _, r in slice_df.iterrows()
    ]

    y = slice_df["true_direction"].astype(int)
    print("\n=== Realisable configurations on harmonized slice ===")
    out_rows = []
    for name, col in [("A1 TFT only", "pred_A1"),
                      ("A4 TFT+Graph (gated)", "pred_A4"),
                      ("A8 TFT+Graph (no gate)", "pred_A8"),
                      ("Graph-only", "pred_graph_only")]:
        p = slice_df[col].astype(int)
        acc, lo, hi = boot_ci(y, p)
        mf1 = macro_f1(y, p)
        chi2, pv = mcnemar(y, slice_df["pred_A1"].astype(int), p)
        flips = int((p != slice_df["pred_A1"].astype(int)).sum())
        print(f"  {name:25s} acc={acc:6.2f}%  CI=[{lo:.2f},{hi:.2f}]  macroF1={mf1:.3f}  McN vs A1 p={pv:.4f}  flips={flips}")
        out_rows.append({"config": name, "acc": acc, "ci_lo": lo, "ci_hi": hi,
                         "macro_f1": mf1, "mcnemar_vs_A1_chi2": chi2,
                         "mcnemar_vs_A1_p": pv, "flips_vs_A1": flips})

    print("\n=== Weight sensitivity: w_graph in {0, 0.20, 0.37, 0.50, 0.65} ===")
    sweep = []
    for w_g_share in [0.0, 0.20, 0.37, 0.50, 0.65]:
        w_t_share = 1 - w_g_share

        def _combine(row):
            return combine_orchestrator(row["pred_tft_v3"], row["graph_score"], row["graph_conf"],
                                        w_t_share, w_g_share, gate_active=True)

        p = slice_df.apply(_combine, axis=1).astype(int)
        acc, lo, hi = boot_ci(y, p)
        chi2, pv = mcnemar(y, slice_df["pred_A1"].astype(int), p)
        flips = int((p != slice_df["pred_A1"].astype(int)).sum())
        print(f"  w_graph={w_g_share:.2f}  acc={acc:.2f}% [{lo:.2f},{hi:.2f}]  McN p={pv:.4f}  flips={flips}")
        sweep.append({"w_graph": w_g_share, "acc": acc, "ci_lo": lo, "ci_hi": hi,
                      "mcnemar_vs_A1_p": pv, "flips_vs_A1": flips})

    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    slice_df[["ticker", "date", "true_direction", "pred_tft_v3",
              "graph_score", "graph_conf", "pred_graph_only",
              "pred_A4", "pred_A8"]].to_csv(PRED_OUT, index=False)
    print(f"\nPer-row CSV: {PRED_OUT}")

    summary = {
        "n": int(len(slice_df)),
        "up_ratio": float(y.mean()),
        "graph_proxy": {"window_days": WINDOW, "top_k": TOP_K, "min_corr": MIN_CORR},
        "graph_coverage": {
            "rows_with_neighbors": covered,
            "pct": float(100 * covered / len(slice_df)),
            "rows_passing_orchestrator_gate": passing_gate,
        },
        "configs": out_rows,
        "weight_sensitivity": sweep,
        "data_dependency": {
            "offline_realizable": ["A1", "A4_gated", "A4_nogate_A8", "graph_only"],
            "requires_postgres": ["A2 (fundamental)", "A3 (sentiment)", "A5 (fund+sent)",
                                  "A6 (seven-agent)", "A7 (no DQ rules)", "A9 (no weight redist)"],
            "reason": ("FundamentalAgent depends on financial_ratios table + trained XGBoost "
                       "checkpoint; SentimentAgent depends on news_articles pre-scored table. "
                       "Neither is shipped offline."),
        },
    }
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary    : {SUMMARY_OUT}")
    print(f"\nElapsed: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
