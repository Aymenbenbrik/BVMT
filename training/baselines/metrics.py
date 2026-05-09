"""training/baselines/metrics.py

Reusable metrics framework for Phase 1 of paper/improvement_plan.md
(Action 1.5). Produces every number the new tab:baselines and
tab:per_stock_breakdown will hold.

Pure-Python + numpy + (optionally) scipy. No torch / sklearn / pandas
dependency in the metric core, so the file can be imported from the
trivial baselines, the seed-sweep aggregator, the per-stock script,
and any future ML baseline alike.

Conventions
    y_true and y_pred are 1-D arrays of {0, 1} where 1 = UP and 0 = DOWN.
    All "accuracy" values are returned in [0, 1] (multiply by 100 for
    percentage display); F1 likewise.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np


def as_int_array(x: Iterable[Any]) -> np.ndarray:
    return np.asarray(list(x), dtype=int)


# ----------------------------------------------------------------------
# Single-model metrics
# ----------------------------------------------------------------------

def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    yt = as_int_array(y_true)
    yp = as_int_array(y_pred)
    if yt.size == 0:
        return float("nan")
    return float((yt == yp).mean())


def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, int]:
    yt = as_int_array(y_true)
    yp = as_int_array(y_pred)
    return {
        "tn": int(((yt == 0) & (yp == 0)).sum()),
        "fp": int(((yt == 0) & (yp == 1)).sum()),
        "fn": int(((yt == 1) & (yp == 0)).sum()),
        "tp": int(((yt == 1) & (yp == 1)).sum()),
    }


def per_class_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, dict[str, float]]:
    cm = confusion_matrix(y_true, y_pred)
    out = {}
    for cls, name in ((1, "UP"), (0, "DOWN")):
        if cls == 1:
            tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
        else:
            tp, fp, fn = cm["tn"], cm["fn"], cm["fp"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    out["macro_f1"] = round((out["UP"]["f1"] + out["DOWN"]["f1"]) / 2.0, 4)
    return out


# ----------------------------------------------------------------------
# Bootstrap confidence intervals
# ----------------------------------------------------------------------

def bootstrap_accuracy_ci(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    n_resamples: int = 1000,
    ci: float = 0.95,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the accuracy."""
    yt = as_int_array(y_true)
    yp = as_int_array(y_pred)
    n = yt.size
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    correct = (yt == yp).astype(np.float32)
    boots = rng.choice(correct, size=(n_resamples, n), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1 - alpha))
    return lo, hi


# ----------------------------------------------------------------------
# Pairwise tests for two models on the same paired predictions
# ----------------------------------------------------------------------

def mcnemar_pvalue(
    y_true: Iterable[int],
    y_pred_a: Iterable[int],
    y_pred_b: Iterable[int],
) -> dict[str, Any]:
    """McNemar test on two classifiers with identical input set.

    Returns the contingency cells b (a wrong, b right), c (a right,
    b wrong) and a chi-square p-value with continuity correction
    (Edwards 1948). When b + c < 25, falls back to the exact binomial
    p-value.
    """
    yt = as_int_array(y_true)
    a = as_int_array(y_pred_a)
    b = as_int_array(y_pred_b)
    if not (yt.size == a.size == b.size):
        raise ValueError("y_true, y_pred_a, y_pred_b must have equal length")
    a_correct = a == yt
    b_correct = b == yt
    n_b = int((~a_correct & b_correct).sum())   # a wrong, b right
    n_c = int((a_correct & ~b_correct).sum())   # a right, b wrong
    total_disagree = n_b + n_c
    if total_disagree == 0:
        return {"b": 0, "c": 0, "stat": 0.0, "p_value": 1.0, "test": "trivial"}
    if total_disagree < 25:
        # Exact two-sided binomial test against p = 0.5
        k = min(n_b, n_c)
        p = 0.0
        for i in range(0, k + 1):
            p += math.comb(total_disagree, i) * 0.5 ** total_disagree
        p_value = min(1.0, 2 * p)
        return {"b": n_b, "c": n_c, "stat": float(k), "p_value": float(p_value), "test": "exact_binomial"}
    chi2 = (abs(n_b - n_c) - 1) ** 2 / total_disagree
    # survival of chi-square with 1 dof: p = exp(-chi2 / 2)
    p_value = math.exp(-chi2 / 2)
    return {"b": n_b, "c": n_c, "stat": float(chi2), "p_value": float(p_value), "test": "chi2_continuity"}


# ----------------------------------------------------------------------
# Per-stock breakdown
# ----------------------------------------------------------------------

def per_stock_summary(
    tickers: Iterable[Any],
    y_true: Iterable[int],
    y_pred: Iterable[int],
    bootstrap_n: int = 1000,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Per-ticker accuracy plus an aggregate spread."""
    yt = as_int_array(y_true)
    yp = as_int_array(y_pred)
    tk = np.asarray(list(tickers))
    if not (yt.size == yp.size == tk.size):
        raise ValueError("tickers, y_true and y_pred must have equal length")

    rows = []
    for ticker in np.unique(tk):
        mask = tk == ticker
        sub_t = yt[mask]
        sub_p = yp[mask]
        if sub_t.size == 0:
            continue
        acc = float((sub_t == sub_p).mean())
        lo, hi = bootstrap_accuracy_ci(sub_t, sub_p, n_resamples=bootstrap_n, rng_seed=rng_seed)
        always_down_acc = float((sub_t == 0).mean())
        rows.append({
            "ticker":         str(ticker),
            "n":              int(sub_t.size),
            "n_up":           int((sub_t == 1).sum()),
            "n_down":         int((sub_t == 0).sum()),
            "accuracy":       round(acc, 4),
            "ci_lo":          round(lo, 4),
            "ci_hi":          round(hi, 4),
            "vs_always_down": round(acc - always_down_acc, 4),
        })

    rows.sort(key=lambda r: r["accuracy"], reverse=True)
    if rows:
        accs = np.array([r["accuracy"] for r in rows])
        spread = {
            "mean":             round(float(accs.mean()), 4),
            "median":           round(float(np.median(accs)), 4),
            "std":              round(float(accs.std(ddof=1)) if accs.size > 1 else 0.0, 4),
            "q1":               round(float(np.quantile(accs, 0.25)), 4),
            "q3":               round(float(np.quantile(accs, 0.75)), 4),
            "min":              round(float(accs.min()), 4),
            "max":              round(float(accs.max()), 4),
            "n_tickers":        int(len(rows)),
            "frac_below_50":    round(float((accs < 0.50).mean()), 4),
            "frac_below_581":   round(float((accs < 0.581).mean()), 4),
            "frac_above_77_4":  round(float((accs >= 0.774).mean()), 4),
        }
    else:
        spread = {}
    return {"rows": rows, "spread": spread}


# ----------------------------------------------------------------------
# Single-model summary helper (everything in one call)
# ----------------------------------------------------------------------

def evaluate_model(
    name: str,
    y_true: Iterable[int],
    y_pred: Iterable[int],
    tickers: Iterable[Any] | None = None,
    bootstrap_n: int = 1000,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Single-call helper that returns the full metric block for one model."""
    yt = as_int_array(y_true)
    yp = as_int_array(y_pred)
    n = yt.size
    acc = accuracy(yt, yp)
    lo, hi = bootstrap_accuracy_ci(yt, yp, n_resamples=bootstrap_n, rng_seed=rng_seed)
    cm = confusion_matrix(yt, yp)
    per_class = per_class_metrics(yt, yp)
    summary: dict[str, Any] = {
        "name":             name,
        "n":                int(n),
        "accuracy":         round(acc, 4),
        "ci95_lo":          round(lo, 4),
        "ci95_hi":          round(hi, 4),
        "confusion_matrix": cm,
        "per_class":        per_class,
    }
    if tickers is not None:
        summary["per_stock"] = per_stock_summary(tickers, yt, yp,
                                                 bootstrap_n=bootstrap_n,
                                                 rng_seed=rng_seed)
    return summary


# ----------------------------------------------------------------------
# Cross-baseline pairwise table
# ----------------------------------------------------------------------

def pairwise_mcnemar_table(
    y_true: Iterable[int],
    predictions: dict[str, Iterable[int]],
) -> list[dict[str, Any]]:
    """Run McNemar test on every ordered pair of models and return a flat list."""
    names = list(predictions.keys())
    yt = as_int_array(y_true)
    out = []
    for i, a_name in enumerate(names):
        for j, b_name in enumerate(names):
            if i >= j:
                continue
            res = mcnemar_pvalue(yt, predictions[a_name], predictions[b_name])
            out.append({
                "a": a_name,
                "b": b_name,
                "b_wrong_a_right": res["c"],
                "a_wrong_b_right": res["b"],
                "stat": round(res["stat"], 4),
                "p_value": round(res["p_value"], 6),
                "test": res["test"],
            })
    return out
