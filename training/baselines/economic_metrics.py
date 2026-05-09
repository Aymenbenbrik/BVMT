"""training/baselines/economic_metrics.py

Action 1.6 of paper/improvement_plan.md: economic / backtest metrics
for every directional baseline persisted under results/baselines/.
Without this layer, the article only reports raw directional accuracy
-- the supervisor critique correctly noted that on a thin market a
high-accuracy classifier can still be unprofitable once spread,
volume, and execution friction are taken into account.

INPUTS
    A predictions CSV with columns:
        ticker, date, true_direction, pred_<modelname>, ...
    All non-numeric pred_ columns are interpreted as binary direction
    in {0=DOWN, 1=UP}.

    The 2025 7-day forward return is reconstructed from close_price by
    a per-ticker shift(-7), exactly as train_tft_v3.py does internally.
    The same boundary-drop discipline of Section 5b applies (rows whose
    target row would fall past the test window are excluded).

STRATEGY
    Two side-of-trade conventions, both starting at close_price[t] and
    closing 7 trading days later:

    long_short  : pred=1 -> +1 unit long; pred=0 -> -1 unit short.
                  Yields per-row PnL = (2*pred - 1) * future_return_7d.

    long_only   : pred=1 -> +1 unit long; pred=0 -> flat (0).
                  Yields per-row PnL = pred * future_return_7d.

    Per-row PnL is then aggregated to a daily PnL series by averaging
    all signals active on day d (equivalent to equal-weighted exposure
    across active tickers).

TRANSACTION COSTS
    A configurable round-trip cost (default 50 bps = 0.005) is
    subtracted on every flip (long_only) or on every trade (long_short).
    Long-only with the always-UP rule pays the cost only on the first
    trade; the always-DOWN rule under long-only is permanently flat
    and pays nothing -- this is informative, not a bug.

METRICS
    cumulative_return       sum of daily log-equiv returns
    annualised_return       (1+mean_daily)^252 - 1
    sharpe_ratio            sqrt(252) * (mean - rf_daily) / std
    sortino_ratio           sqrt(252) * (mean - rf_daily) / downside_std
    max_drawdown            worst peak-to-trough cumulative drawdown
    profit_factor           sum_wins / |sum_losses|
    hit_ratio               share of days with positive PnL
    deflated_sharpe_p       p-value that the observed Sharpe is real
                            against the inflation expected when
                            multiple strategies are tested
                            (Bailey & Lopez de Prado 2014).

USAGE
    python -m training.baselines.economic_metrics \\
        --features data/features/tft_features.csv \\
        --predictions results/baselines/trivial_predictions.csv \\
                       results/baselines/ml_classics_predictions.csv \\
        --tc 0.005 --rf 0.08 \\
        --out results/baselines/economic_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

VAL_END = "2025-01-01"
HORIZON = 7  # matches train_tft_v3.py
TRADING_DAYS_PER_YEAR = 252


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=ROOT / "data" / "features" / "tft_features.csv")
    p.add_argument("--predictions", type=Path, nargs="+",
                   default=[
                       ROOT / "results" / "baselines" / "trivial_predictions.csv",
                       ROOT / "results" / "baselines" / "ml_classics_predictions.csv",
                   ])
    p.add_argument("--out", type=Path, default=ROOT / "results" / "baselines" / "economic_summary.json")
    p.add_argument("--tc", type=float, default=0.005,
                   help="Round-trip transaction cost as a fraction (default 0.005 = 50 bps).")
    p.add_argument("--rf", type=float, default=0.08,
                   help="Annual risk-free rate (default 0.08, BCT TMM proxy).")
    p.add_argument("--strategy", choices=["long_short", "long_only", "both"], default="both")
    return p.parse_args()


# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------

def reconstruct_forward_returns(features_path: Path) -> pd.DataFrame:
    """Compute per-row future_return_7d from close_price exactly as
    train_tft_v3 does. Returns a DataFrame indexed by (ticker, date)
    with columns close_price and future_return_7d."""
    df = pd.read_csv(features_path, usecols=["ticker", "date", "time_idx", "close_price"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "time_idx"]).reset_index(drop=True)
    df["future_close"] = df.groupby("ticker")["close_price"].shift(-HORIZON)
    df["future_return_7d"] = (
        (df["future_close"] - df["close_price"])
        / df["close_price"].replace(0, np.nan)
    ).astype("float32")
    df = df.dropna(subset=["future_return_7d"])
    df = df[["ticker", "date", "close_price", "future_return_7d"]].copy()
    return df


def load_predictions(paths: list[Path], returns: pd.DataFrame) -> pd.DataFrame:
    """Concatenate every predictions CSV horizontally on (ticker, date),
    drop rows whose forward return cannot be reconstructed."""
    base = returns[returns["date"] >= pd.Timestamp(VAL_END)].copy()
    base["date"] = pd.to_datetime(base["date"])
    merged = base.copy()
    pred_cols: list[str] = []

    for path in paths:
        if not path.exists():
            print(f"WARN: predictions file not found, skipping: {path}")
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        cols = [c for c in df.columns if c.startswith("pred_")]
        if not cols:
            print(f"WARN: no pred_* columns in {path}, skipping")
            continue
        df = df[["ticker", "date", *cols]].copy()
        merged = merged.merge(df, on=["ticker", "date"], how="left")
        pred_cols.extend(cols)

    pred_cols = list(dict.fromkeys(pred_cols))  # de-duplicate, keep order
    if not pred_cols:
        raise SystemExit("No prediction columns found across all CSVs.")

    # Drop rows with no model prediction at all.
    merged = merged.dropna(subset=pred_cols, how="all").reset_index(drop=True)
    return merged, pred_cols


# ----------------------------------------------------------------------
# Strategies and per-row PnL
# ----------------------------------------------------------------------

def per_row_pnl(pred: pd.Series, ret: pd.Series, strategy: str) -> pd.Series:
    """Per-row pseudo-PnL before transaction costs."""
    p = pd.to_numeric(pred, errors="coerce")
    if strategy == "long_short":
        side = 2 * p - 1                    # 1 -> +1, 0 -> -1
    elif strategy == "long_only":
        side = p                            # 1 -> +1, 0 -> 0
    else:
        raise ValueError(strategy)
    return side * ret


def aggregate_to_daily(rows: pd.DataFrame, value_col: str) -> pd.Series:
    """Equal-weighted daily PnL across all signals issued that day."""
    daily = rows.groupby("date")[value_col].mean().sort_index()
    return daily


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def annualised_return(daily: pd.Series) -> float:
    if daily.empty:
        return float("nan")
    mu = float(daily.mean())
    return float((1 + mu) ** TRADING_DAYS_PER_YEAR - 1)


def sharpe_ratio(daily: pd.Series, rf_annual: float) -> float:
    if daily.empty or daily.std() == 0:
        return float("nan")
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = daily - rf_daily
    return float(math.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess.std(ddof=1))


def sortino_ratio(daily: pd.Series, rf_annual: float) -> float:
    if daily.empty:
        return float("nan")
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = daily - rf_daily
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return float("inf") if excess.mean() > 0 else float("nan")
    return float(math.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / downside.std(ddof=1))


def max_drawdown(daily: pd.Series) -> float:
    if daily.empty:
        return float("nan")
    equity = (1 + daily).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def profit_factor(daily: pd.Series) -> float:
    if daily.empty:
        return float("nan")
    wins = daily[daily > 0].sum()
    losses = -daily[daily < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def hit_ratio(daily: pd.Series) -> float:
    if daily.empty:
        return float("nan")
    return float((daily > 0).mean())


def cumulative_return(daily: pd.Series) -> float:
    if daily.empty:
        return float("nan")
    return float(((1 + daily).prod() - 1))


# ----------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014, simplified)
#
# Given an observed Sharpe SR_obs, the standard deviation of SR is:
#     se(SR) = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (T-1))
# The expected maximum Sharpe under the null when M independent
# strategies are tested is:
#     E[max SR] = sqrt(2 * ln(M)) - (gamma + ln(ln(M))) / sqrt(2*ln(M))
# DSR is then the probability that SR_obs exceeds the expected maximum:
#     DSR = Phi((SR_obs - E[max SR]) / se(SR))
# We pass M (number of trials -- here, number of baselines tested).
# ----------------------------------------------------------------------

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe_under_null(M: int) -> float:
    """E[max SR_i] when M iid strategies have true SR=0 and unit
    variance. Closed-form approximation from Bailey & Lopez de Prado."""
    M = max(M, 2)
    log_m = math.log(M)
    if log_m <= 0:
        return 0.0
    return float(math.sqrt(2 * log_m) - (EULER_MASCHERONI + math.log(log_m)) / math.sqrt(2 * log_m))


def deflated_sharpe_pvalue(
    daily: pd.Series,
    rf_annual: float,
    n_trials: int,
) -> dict:
    """Return DSR, expected-max-Sharpe, and the implied p-value."""
    if daily.empty or daily.std() == 0:
        return {"dsr": float("nan"), "expected_max_sr": float("nan"), "p_value": float("nan"), "n_trials": n_trials}

    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = daily - rf_daily
    sr = math.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess.std(ddof=1)

    # Higher moments of daily excess returns (do not annualise here)
    daily_excess = excess.dropna()
    n = len(daily_excess)
    if n < 5:
        return {"dsr": float("nan"), "expected_max_sr": float("nan"), "p_value": float("nan"), "n_trials": n_trials}

    skew = float(((daily_excess - daily_excess.mean()) ** 3).mean() / (daily_excess.std() ** 3))
    kurt = float(((daily_excess - daily_excess.mean()) ** 4).mean() / (daily_excess.std() ** 4))

    # SR standard error in annual units
    sr_daily = sr / math.sqrt(TRADING_DAYS_PER_YEAR)
    var_sr_daily = max((1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily ** 2) / max(n - 1, 1), 1e-12)
    se_sr = math.sqrt(var_sr_daily) * math.sqrt(TRADING_DAYS_PER_YEAR)

    e_max = expected_max_sharpe_under_null(n_trials)
    z = (sr - e_max) / max(se_sr, 1e-12)
    dsr = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return {
        "dsr": round(float(dsr), 6),
        "expected_max_sr": round(float(e_max), 4),
        "se_sr": round(float(se_sr), 4),
        "skew": round(skew, 4),
        "kurt": round(kurt, 4),
        "n_obs": int(n),
        "n_trials": int(n_trials),
        "p_value": round(float(1 - dsr), 6),
    }


# ----------------------------------------------------------------------
# Per-model summary
# ----------------------------------------------------------------------

def evaluate_strategy(
    df: pd.DataFrame,
    pred_col: str,
    strategy: str,
    tc: float,
    rf_annual: float,
    n_trials: int,
) -> dict | None:
    sub = df[["date", "future_return_7d", pred_col]].dropna(subset=[pred_col]).copy()
    if sub.empty:
        return None

    pnl_gross = per_row_pnl(sub[pred_col], sub["future_return_7d"], strategy)
    sub["pnl_gross"] = pnl_gross
    # Transaction-cost approximation: we charge tc on every entry where
    # the position is non-zero. Long-only with pred=0 keeps cash and
    # incurs no cost on those rows; long_short always trades.
    if strategy == "long_only":
        sub["pnl_net"] = sub["pnl_gross"] - tc * (sub[pred_col].astype(float) != 0).astype(float)
    else:
        sub["pnl_net"] = sub["pnl_gross"] - tc

    daily_gross = aggregate_to_daily(sub, "pnl_gross")
    daily_net = aggregate_to_daily(sub, "pnl_net")

    out = {
        "strategy": strategy,
        "n_trades": int(len(sub)),
        "n_days": int(daily_net.size),
        "gross": {
            "cumulative_return": round(cumulative_return(daily_gross), 4),
            "annualised_return": round(annualised_return(daily_gross), 4),
            "sharpe_ratio": round(sharpe_ratio(daily_gross, rf_annual), 4),
            "sortino_ratio": round(sortino_ratio(daily_gross, rf_annual), 4),
            "max_drawdown": round(max_drawdown(daily_gross), 4),
            "profit_factor": round(profit_factor(daily_gross), 4),
            "hit_ratio": round(hit_ratio(daily_gross), 4),
            "deflated_sharpe": deflated_sharpe_pvalue(daily_gross, rf_annual, n_trials),
        },
        "net": {
            "cumulative_return": round(cumulative_return(daily_net), 4),
            "annualised_return": round(annualised_return(daily_net), 4),
            "sharpe_ratio": round(sharpe_ratio(daily_net, rf_annual), 4),
            "sortino_ratio": round(sortino_ratio(daily_net, rf_annual), 4),
            "max_drawdown": round(max_drawdown(daily_net), 4),
            "profit_factor": round(profit_factor(daily_net), 4),
            "hit_ratio": round(hit_ratio(daily_net), 4),
            "deflated_sharpe": deflated_sharpe_pvalue(daily_net, rf_annual, n_trials),
        },
    }
    return out


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def render_latex_rows(per_model: dict, strategy: str, version: str) -> list[str]:
    """One LaTeX row per model for tab:economic_metrics."""
    rows = []
    for name, by_strat in per_model.items():
        s = by_strat.get(strategy)
        if s is None:
            continue
        m = s[version]
        sharpe = m["sharpe_ratio"]
        cum = m["cumulative_return"]
        ar = m["annualised_return"]
        mdd = m["max_drawdown"]
        pf = m["profit_factor"]
        hr = m["hit_ratio"]
        ds_p = m["deflated_sharpe"]["p_value"]
        rows.append(
            f"{name.replace('pred_', '').replace('_', ' '):<22} & "
            f"{cum*100:6.2f}\\% & {ar*100:6.2f}\\% & "
            f"{sharpe:6.2f} & {mdd*100:6.2f}\\% & "
            f"{pf:5.2f} & {hr*100:5.1f}\\% & "
            f"{ds_p:.3f} \\\\"
        )
    return rows


def main() -> int:
    args = parse_args()

    print(f"Reading features and reconstructing 7-day forward returns from {args.features} ...")
    returns = reconstruct_forward_returns(args.features)
    print(f"  Forward returns table: {len(returns):,} rows")

    print("\nLoading predictions ...")
    df, pred_cols = load_predictions(args.predictions, returns)
    print(f"  Joined predictions: {len(df):,} rows, {len(pred_cols)} models")
    n_trials = max(len(pred_cols), 2)

    strategies = ["long_short", "long_only"] if args.strategy == "both" else [args.strategy]
    per_model: dict[str, dict] = {}

    print(f"\nTransaction cost: {args.tc*100:.2f} bps round-trip = {args.tc:.4f}")
    print(f"Risk-free rate  : {args.rf*100:.1f}% annual")
    print(f"Trials for DSR  : {n_trials}\n")

    for col in pred_cols:
        per_model[col] = {}
        for strat in strategies:
            res = evaluate_strategy(df, col, strat, tc=args.tc, rf_annual=args.rf, n_trials=n_trials)
            if res is None:
                continue
            per_model[col][strat] = res

    # Console summary
    for strat in strategies:
        for version, label in (("net", "NET (with TC)"), ("gross", "GROSS (no TC)")):
            print(f"=== Strategy: {strat}  --  {label} ===")
            print(f"{'Model':<24} {'cum%':>7} {'ann%':>7} {'Sharpe':>7} {'maxDD%':>8} {'PF':>5} {'hit%':>6} {'DSR p':>7}")
            print("-" * 78)
            for name, by_strat in per_model.items():
                s = by_strat.get(strat)
                if s is None:
                    continue
                m = s[version]
                print(
                    f"{name:<24} "
                    f"{m['cumulative_return']*100:>6.2f}% "
                    f"{m['annualised_return']*100:>6.2f}% "
                    f"{m['sharpe_ratio']:>7.2f} "
                    f"{m['max_drawdown']*100:>7.2f}% "
                    f"{m['profit_factor']:>5.2f} "
                    f"{m['hit_ratio']*100:>5.1f}% "
                    f"{m['deflated_sharpe']['p_value']:>7.4f}"
                )
            print()

    # JSON output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_obj = {
        "config": {
            "transaction_cost": args.tc,
            "risk_free_annual": args.rf,
            "n_trials_for_dsr": n_trials,
            "horizon_days": HORIZON,
            "trading_days_per_year": TRADING_DAYS_PER_YEAR,
            "strategies": strategies,
            "n_predictions_total": int(len(df)),
        },
        "per_model": per_model,
        "latex_rows": {
            f"{strat}_{v}": render_latex_rows(per_model, strat, v)
            for strat in strategies for v in ("gross", "net")
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o))
    print(f"Wrote: {args.out}")

    print("\nLaTeX rows (long_short, NET) -- paste into tab:economic_metrics:")
    for line in out_obj["latex_rows"].get("long_short_net", []):
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
