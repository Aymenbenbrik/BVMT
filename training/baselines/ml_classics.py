"""training/baselines/ml_classics.py

Action 1.2 of paper/improvement_plan.md: tabular ML baselines on the
flattened 60-day x 15-feature window. The article's tab:baselines
will gain three rows from this script:

    Logistic Regression (L2) -- linear baseline, with C grid
    Random Forest            -- non-parametric tree ensemble, with
                                max_depth grid
    XGBoost (technical)      -- gradient boosting on the same input

Each baseline is selected on the 2024 validation split and evaluated
on the 2025 test split. Per-row predictions are persisted next to the
trivial-baseline outputs so the McNemar pipeline (Action 1.5b/1.8)
can compare every model on identical test rows.

ANTI-LEAK DISCIPLINE
    - Per-stock standardisation: mean/std are fit on the train rows
      only, then applied to validation and test. No statistic ever
      crosses the 2024-01-01 boundary.
    - Boundary drop: rows in train whose target row falls in val are
      removed (Section 5b of train_tft_v3.py). Same for val/test.
    - The flattened window for date t uses only feature rows
      [t - W + 1, ..., t]; the target direction_7d at t already encodes
      the future return so it is a pure label leak guard, not a
      feature.

USAGE
    python -m training.baselines.ml_classics \\
        --window 60 \\
        --out    results/baselines/

Smoke test on a few stocks first (~30 s):
    python -m training.baselines.ml_classics --smoke 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from training.baselines.metrics import (  # noqa: E402
    evaluate_model,
    pairwise_mcnemar_table,
)

TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"
HORIZON = 7  # matches direction_7d

FEATURES = [
    "close_price", "open_price", "high_price", "low_price",
    "volume", "daily_return", "ma_5", "ma_20", "ma_50",
    "rsi_14", "volatility_20", "volume_ma_20",
    "price_to_ma20", "rsi_momentum", "volume_spike",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "features" / "tft_features.csv")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "baselines")
    p.add_argument("--window", type=int, default=60,
                   help="Flattened window size (must match what CNN-LSTM consumes).")
    p.add_argument("--smoke", type=int, default=0,
                   help="If > 0, restrict to the first K tickers for a quick test run.")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rf-max-depth", type=str, default="5,10,None",
                   help="Comma list for RF max_depth grid. Use 'None' for unbounded.")
    p.add_argument("--logreg-C", type=str, default="0.01,0.1,1,10")
    p.add_argument("--rf-n-estimators", type=int, default=300)
    return p.parse_args()


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Three derived features named in the article (Section III.B)."""
    if "price_to_ma20" not in df.columns:
        df["price_to_ma20"] = (df["close_price"] / df["ma_20"].replace(0, np.nan)).fillna(1.0).clip(0.5, 2.0).astype("float32")
    if "rsi_momentum" not in df.columns:
        df["rsi_momentum"] = (
            df.groupby("ticker")["rsi_14"]
              .transform(lambda x: x - x.shift(5))
              .fillna(0.0).astype("float32")
        )
    if "volume_spike" not in df.columns:
        df["volume_spike"] = (df["volume"] / df["volume_ma_20"].replace(0, np.nan)).fillna(1.0).clip(0.0, 10.0).astype("float32")
    return df


def fit_per_stock_stats(train_df: pd.DataFrame) -> dict[str, dict[str, tuple[float, float]]]:
    """Per-stock mean/std fit on the train period only."""
    stats: dict[str, dict[str, tuple[float, float]]] = {}
    for ticker, g in train_df.groupby("ticker"):
        s = {}
        for f in FEATURES:
            v = g[f].astype("float32")
            mu = float(v.mean())
            sd = float(v.std())
            if not np.isfinite(sd) or sd < 1e-8:
                sd = 1.0
            s[f] = (mu, sd)
        stats[str(ticker)] = s
    return stats


def apply_per_stock_norm(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    # Vectorised per-ticker normalisation
    for f in FEATURES:
        mus = df["ticker"].map(lambda t: stats.get(str(t), {}).get(f, (0.0, 1.0))[0]).astype("float32")
        sds = df["ticker"].map(lambda t: stats.get(str(t), {}).get(f, (0.0, 1.0))[1]).astype("float32")
        df[f] = ((df[f].astype("float32") - mus) / sds).astype("float32")
    return df


def build_windows(
    df: pd.DataFrame,
    window: int,
    split_lo: pd.Timestamp,
    split_hi: pd.Timestamp,
    boundary_drop: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For every ticker, slide a window of size `window` ending at date t and
    label it with direction_7d at t. Keep windows whose date is in
    [split_lo, split_hi). Boundary-drop: skip windows whose target row
    (date + ~7 trading days) crosses split_hi."""
    Xs, ys, tickers, dates = [], [], [], []
    for ticker, g in df.groupby("ticker", sort=True):
        g = g.sort_values("time_idx").reset_index(drop=True)
        feats = g[FEATURES].astype("float32").to_numpy()
        labels = g["direction_7d"].astype(int).to_numpy()
        d = pd.to_datetime(g["date"]).to_numpy()
        if len(g) < window + HORIZON:
            continue
        n = len(g)
        for t in range(window - 1, n):
            row_date = d[t]
            if not (split_lo <= row_date < split_hi):
                continue
            if boundary_drop:
                # The target at row t depends on prices through t+HORIZON;
                # if that target row sits past split_hi, drop the sample to
                # avoid pulling labels from the next split.
                target_idx = t + HORIZON
                if target_idx >= n:
                    continue
                if d[target_idx] >= split_hi:
                    continue
            Xs.append(feats[t - window + 1: t + 1].reshape(-1))
            ys.append(int(labels[t]))
            tickers.append(str(ticker))
            dates.append(np.datetime_as_string(row_date, unit="D"))
    if not Xs:
        return (np.empty((0, len(FEATURES) * window), dtype="float32"),
                np.empty(0, dtype=int), np.empty(0, dtype=object),
                np.empty(0, dtype=object))
    return (
        np.stack(Xs).astype("float32"),
        np.asarray(ys, dtype=int),
        np.asarray(tickers, dtype=object),
        np.asarray(dates, dtype=object),
    )


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

def fit_logreg(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, C_grid: list[float]) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    best = None
    for C in C_grid:
        clf = LogisticRegression(
            penalty="l2", C=C, solver="lbfgs", max_iter=2000,
            class_weight="balanced", n_jobs=1,
        )
        clf.fit(X_tr, y_tr)
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        if best is None or val_acc > best["val_acc"]:
            best = {"C": C, "val_acc": float(val_acc), "model": clf}
    return best


def fit_random_forest(X_tr, y_tr, X_val, y_val, depth_grid, n_estimators) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    best = None
    for d in depth_grid:
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=d,
            n_jobs=-1, random_state=0, class_weight="balanced",
        )
        clf.fit(X_tr, y_tr)
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        if best is None or val_acc > best["val_acc"]:
            best = {"max_depth": "None" if d is None else d, "val_acc": float(val_acc), "model": clf}
    return best


def fit_xgboost(X_tr, y_tr, X_val, y_val) -> dict:
    import xgboost as xgb
    pos = max(int((y_tr == 1).sum()), 1)
    neg = max(int((y_tr == 0).sum()), 1)
    spw = neg / pos
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=spw, tree_method="hist",
        n_jobs=-1, random_state=0,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    from sklearn.metrics import accuracy_score
    return {
        "n_estimators": 400, "max_depth": 6, "lr": 0.05, "scale_pos_weight": round(spw, 3),
        "val_acc": float(accuracy_score(y_val, clf.predict(X_val))), "model": clf,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "time_idx"]).reset_index(drop=True)
    df = add_relative_features(df)

    if args.smoke > 0:
        keep = sorted(df["ticker"].unique())[: args.smoke]
        df = df[df["ticker"].isin(keep)].copy()
        print(f"  Smoke mode: restricted to {len(keep)} tickers ({keep})")

    train_lo = pd.Timestamp("1900-01-01")
    train_hi = pd.Timestamp(TRAIN_END)
    val_lo = pd.Timestamp(TRAIN_END)
    val_hi = pd.Timestamp(VAL_END)
    test_lo = pd.Timestamp(VAL_END)
    test_hi = pd.Timestamp("2099-01-01")

    train_rows = df[df["date"] < train_hi].copy()
    print(f"\nFitting per-stock normalisation on {len(train_rows):,} train rows ({train_rows['ticker'].nunique()} tickers).")
    stats = fit_per_stock_stats(train_rows)
    df_norm = apply_per_stock_norm(df, stats)

    print(f"Building windows (size={args.window})...")
    t0 = time.time()
    X_tr, y_tr, tk_tr, _ = build_windows(df_norm, args.window, train_lo, train_hi)
    X_va, y_va, tk_va, _ = build_windows(df_norm, args.window, val_lo, val_hi)
    X_te, y_te, tk_te, dt_te = build_windows(df_norm, args.window, test_lo, test_hi)
    print(f"  train: {X_tr.shape}  val: {X_va.shape}  test: {X_te.shape}  ({time.time()-t0:.1f}s)")

    if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
        raise SystemExit("Empty train or test split -- check window size / smoke setting.")

    rng_seed = args.seed
    summaries = []
    preds: dict[str, np.ndarray] = {}

    # ---- Logistic Regression ----
    print("\n[1/3] Logistic Regression (L2) -- C grid", args.logreg_C)
    C_grid = [float(c) for c in args.logreg_C.split(",")]
    t0 = time.time()
    lr = fit_logreg(X_tr, y_tr, X_va, y_va, C_grid)
    p_lr = lr["model"].predict(X_te)
    print(f"  best C={lr['C']}  val_acc={lr['val_acc']:.4f}  ({time.time()-t0:.1f}s)")
    summaries.append({
        "model": "logreg_l2",
        "best_hparams": {"C": lr["C"]},
        "val_acc": lr["val_acc"],
        **evaluate_model("logreg_l2", y_te, p_lr, tickers=tk_te,
                         bootstrap_n=args.bootstrap, rng_seed=rng_seed),
    })
    preds["logreg_l2"] = p_lr

    # ---- Random Forest ----
    depth_grid = []
    for d in args.rf_max_depth.split(","):
        d = d.strip()
        depth_grid.append(None if d.lower() == "none" else int(d))
    print(f"\n[2/3] Random Forest -- n_estimators={args.rf_n_estimators}, max_depth grid {depth_grid}")
    t0 = time.time()
    rf = fit_random_forest(X_tr, y_tr, X_va, y_va, depth_grid, args.rf_n_estimators)
    p_rf = rf["model"].predict(X_te)
    print(f"  best max_depth={rf['max_depth']}  val_acc={rf['val_acc']:.4f}  ({time.time()-t0:.1f}s)")
    summaries.append({
        "model": "random_forest",
        "best_hparams": {"max_depth": rf["max_depth"], "n_estimators": args.rf_n_estimators},
        "val_acc": rf["val_acc"],
        **evaluate_model("random_forest", y_te, p_rf, tickers=tk_te,
                         bootstrap_n=args.bootstrap, rng_seed=rng_seed),
    })
    preds["random_forest"] = p_rf

    # ---- XGBoost ----
    print("\n[3/3] XGBoost -- 400 trees, max_depth=6, lr=0.05")
    t0 = time.time()
    xgb_b = fit_xgboost(X_tr, y_tr, X_va, y_va)
    p_xgb = xgb_b["model"].predict(X_te)
    print(f"  scale_pos_weight={xgb_b['scale_pos_weight']}  val_acc={xgb_b['val_acc']:.4f}  ({time.time()-t0:.1f}s)")
    summaries.append({
        "model": "xgboost_technical",
        "best_hparams": {"n_estimators": xgb_b["n_estimators"], "max_depth": xgb_b["max_depth"], "lr": xgb_b["lr"]},
        "val_acc": xgb_b["val_acc"],
        **evaluate_model("xgboost_technical", y_te, p_xgb, tickers=tk_te,
                         bootstrap_n=args.bootstrap, rng_seed=rng_seed),
    })
    preds["xgboost_technical"] = p_xgb

    # ---- Pairwise McNemar across the three ML baselines ----
    pairs = pairwise_mcnemar_table(y_te, preds)

    # ---- Persist per-row predictions for the McNemar/per-stock pipeline ----
    pred_csv = args.out / "ml_classics_predictions.csv"
    pred_df = pd.DataFrame({
        "ticker":         tk_te,
        "date":           dt_te,
        "true_direction": y_te,
        "pred_logreg_l2":      preds["logreg_l2"],
        "pred_random_forest":  preds["random_forest"],
        "pred_xgboost":        preds["xgboost_technical"],
    })
    pred_df.to_csv(pred_csv, index=False)
    print(f"\nWrote per-row predictions: {pred_csv}")

    # ---- Console summary ----
    print("\n" + "=" * 78)
    print(f"{'Model':<20} {'val':>7} {'test':>7} {'ci_lo':>7} {'ci_hi':>7} {'F1_UP':>6} {'F1_DN':>6} {'macro':>6}")
    print("-" * 78)
    for s in summaries:
        pc = s["per_class"]
        print(
            f"{s['model']:<20} {s['val_acc']*100:6.2f}% {s['accuracy']*100:6.2f}% "
            f"{s['ci95_lo']*100:6.2f}% {s['ci95_hi']*100:6.2f}% "
            f"{pc['UP']['f1']:>6.3f} {pc['DOWN']['f1']:>6.3f} {pc['macro_f1']:>6.3f}"
        )
    print("=" * 78)
    print("\nPairwise McNemar:")
    for r in pairs:
        print(f"  {r['a']:<20} vs {r['b']:<20}  stat={r['stat']:.4f}  p={r['p_value']:.6f}  ({r['test']})")

    out_json = args.out / "ml_classics_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "test_set": {
                "n": int(len(y_te)), "n_tickers": int(len(set(tk_te))),
                "up_ratio": round(float((y_te == 1).mean()), 4),
            },
            "models": [{k: v for k, v in s.items() if k != "model"} for s in summaries],  # drop sklearn refs
            "pairwise_mcnemar": pairs,
        }, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o))
    print(f"\nWrote summary JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
