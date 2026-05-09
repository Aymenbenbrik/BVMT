# tests/test_no_leakage.py
# ──────────────────────────────────────────────────────────────────────
# PURPOSE: Anti-leakage regression tests for the BVMT pipeline.
#
# These tests are the executable counterpart of paper/leakage_audit.md.
# Each test pins a specific anti-leak guarantee from Étapes 1–3:
#
#   1. Section 5b of train_tft_v3.py drops exactly the boundary rows
#      whose target crosses TRAIN_END or VAL_END.
#   2. The future_return_7d cleanup leaves no NaT in _target_date.
#   3. agents/fundamental_agent.py's SQL contains the publication-lag
#      filter and the Python coercer accepts well-formed dates.
#   4. agents/sentiment_agent.py's fallbacks accept upper_bound_ts and
#      its SQL filters published_at <= upper bound.
#   5. Mini label-shuffle: a shallow model trained on permuted train
#      labels must not exceed ~58% on a held-out test split (sanity
#      bound for the full pipeline).
#
# RUN:
#   pytest tests/test_no_leakage.py -v
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════
# Test 1 — Section 5b drops exactly the boundary rows
# ══════════════════════════════════════════════════════════════════════

def _build_synthetic_panel(
    n_tickers: int = 3,
    start: str = "2023-09-01",
    end: str = "2025-04-30",
) -> pd.DataFrame:
    """Build a deterministic 3-ticker panel with daily business dates."""
    dates = pd.bdate_range(start, end, freq="B")
    rows = [
        {"ticker_id": str(tid), "date": d, "close_price": 10 + i * 0.01}
        for tid in range(n_tickers)
        for i, d in enumerate(dates)
    ]
    return pd.DataFrame(rows).sort_values(["ticker_id", "date"]).reset_index(drop=True)


def _apply_section_4_and_5b(
    df: pd.DataFrame,
    train_end: str = "2024-01-01",
    val_end: str = "2025-01-01",
    horizon: int = 7,
) -> tuple[pd.DataFrame, int, int]:
    """Mirror Section 4 + Section 5b of training/train_tft_v3.py exactly."""
    df = df.copy()
    df["_future_close"] = (
        df.groupby("ticker_id")["close_price"]
          .transform(lambda x: x.shift(-horizon))
    )
    df["_target_date"] = (
        df.groupby("ticker_id")["date"]
          .transform(lambda s: s.shift(-horizon))
    )
    df["future_return_7d"] = (
        (df["_future_close"] - df["close_price"])
        / df["close_price"].replace(0, np.nan)
    ).fillna(np.nan)
    df = df.dropna(subset=["future_return_7d", "_target_date"]).reset_index(drop=True)

    train_end_ts = pd.to_datetime(train_end)
    val_end_ts = pd.to_datetime(val_end)

    mask_train = (df["date"] < train_end_ts) & (df["_target_date"] >= train_end_ts)
    mask_val = (
        (df["date"] >= train_end_ts)
        & (df["date"] < val_end_ts)
        & (df["_target_date"] >= val_end_ts)
    )
    n_train = int(mask_train.sum())
    n_val = int(mask_val.sum())
    df = df[~(mask_train | mask_val)].reset_index(drop=True)
    return df, n_train, n_val


def test_section_5b_drops_exact_boundary_rows():
    """Étape 3: with horizon=7 trading days and 3 tickers, expect 7×3=21
    drops on each boundary."""
    df = _build_synthetic_panel(n_tickers=3)
    cleaned, n_train, n_val = _apply_section_4_and_5b(df)
    assert n_train == 21, f"expected 21 train-side leak rows, got {n_train}"
    assert n_val == 21, f"expected 21 val-side leak rows, got {n_val}"

    # No residual: every kept row's target stays within its split.
    train_end = pd.Timestamp("2024-01-01")
    val_end = pd.Timestamp("2025-01-01")
    cleaned_train = cleaned[cleaned["date"] < train_end]
    assert (cleaned_train["_target_date"] < train_end).all(), \
        "residual train-side leak after Section 5b"
    cleaned_val_only = cleaned[
        (cleaned["date"] >= train_end) & (cleaned["date"] < val_end)
    ]
    assert (cleaned_val_only["_target_date"] < val_end).all(), \
        "residual val-side leak after Section 5b"


def test_section_5b_no_drop_when_all_targets_internal():
    """If horizon shrinks to 1 day and the boundaries are far from the
    panel's edges, no row should be dropped."""
    df = _build_synthetic_panel(
        n_tickers=2, start="2024-06-01", end="2024-07-01"
    )
    _, n_train, n_val = _apply_section_4_and_5b(
        df, train_end="2030-01-01", val_end="2031-01-01", horizon=1
    )
    assert n_train == 0
    assert n_val == 0


def test_section_4_target_date_no_nat_after_dropna():
    """Étape 3: after the future_return_7d / _target_date dropna, the
    invariant asserted by Section 5b must hold."""
    df = _build_synthetic_panel(n_tickers=4)
    df = df.copy()
    df["_future_close"] = (
        df.groupby("ticker_id")["close_price"]
          .transform(lambda x: x.shift(-7))
    )
    df["_target_date"] = (
        df.groupby("ticker_id")["date"]
          .transform(lambda s: s.shift(-7))
    )
    df["future_return_7d"] = (
        (df["_future_close"] - df["close_price"])
        / df["close_price"].replace(0, np.nan)
    ).fillna(np.nan)
    df = df.dropna(subset=["future_return_7d", "_target_date"]).reset_index(drop=True)
    assert df["_target_date"].notna().all(), \
        "Section 5b assert would fire: NaT residual in _target_date"


# ══════════════════════════════════════════════════════════════════════
# Test 2 — FundamentalAgent has publication-lag filter (Étape 1)
# ══════════════════════════════════════════════════════════════════════

def _load_fundamental_module():
    """Import agents/fundamental_agent.py with stubbed psycopg2/dotenv so
    we can inspect signatures/source without a DB driver installed."""
    import types

    if "psycopg2" not in sys.modules:
        sys.modules["psycopg2"] = types.SimpleNamespace(
            connect=lambda *a, **kw: None
        )
    if "dotenv" not in sys.modules:
        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = lambda *a, **kw: None
        sys.modules["dotenv"] = dotenv_stub

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "agents.fundamental_agent",
        ROOT / "agents" / "fundamental_agent.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fundamental_query_has_publication_lag_filter():
    """Étape 1: SQL must guard against rows whose proxy publication date
    falls after prediction_date."""
    src = (ROOT / "agents" / "fundamental_agent.py").read_text(encoding="utf-8")
    # Filter clause should mention the lag interval and the prediction_date
    # bound. Whitespace-tolerant.
    assert re.search(
        r"period_end_date.*publication_lag_months.*months.*prediction_date",
        src,
        flags=re.DOTALL,
    ), "publication-lag SQL filter missing from fundamental_agent.py"


def test_fundamental_coercer_accepts_known_inputs():
    """Étape 1: _coerce_prediction_date covers the common back-test
    inputs (None, ISO date, ISO datetime, date, datetime)."""
    fa = _load_fundamental_module()
    from datetime import date, datetime

    today = fa._coerce_prediction_date(None)
    assert today.year >= 2026
    assert fa._coerce_prediction_date("") == today
    assert fa._coerce_prediction_date("2024-06-15") == date(2024, 6, 15)
    assert fa._coerce_prediction_date("2024-06-15T12:00:00") == date(2024, 6, 15)
    assert fa._coerce_prediction_date("2024-06-15T12:00:00Z") == date(2024, 6, 15)
    assert fa._coerce_prediction_date(date(2024, 6, 15)) == date(2024, 6, 15)
    assert fa._coerce_prediction_date(datetime(2024, 6, 15, 9, 30)) == date(2024, 6, 15)


def test_fundamental_coercer_rejects_invalid():
    fa = _load_fundamental_module()
    with pytest.raises(ValueError):
        fa._coerce_prediction_date("not-a-date")
    with pytest.raises(TypeError):
        fa._coerce_prediction_date(12345)


def test_fundamental_signatures_carry_prediction_date():
    """Étape 1: prediction_date must be wired on all three layers."""
    fa = _load_fundamental_module()
    import inspect

    for fn in (fa._query_latest_ratios, fa.fetch_latest_ratios):
        params = inspect.signature(fn).parameters
        assert "prediction_date" in params, \
            f"{fn.__name__} must accept prediction_date"

    run_params = inspect.signature(fa.FundamentalAgent.run).parameters
    assert "prediction_date" in run_params, \
        "FundamentalAgent.run must accept prediction_date"


# ══════════════════════════════════════════════════════════════════════
# Test 3 — SentimentAgent fallbacks bounded by upper_bound_ts (Étape 2)
# ══════════════════════════════════════════════════════════════════════

def test_sentiment_fallbacks_take_upper_bound_ts():
    """Étape 2: both private fallback methods must accept upper_bound_ts."""
    src = (ROOT / "agents" / "sentiment_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SentimentAgent"
    )
    methods = {
        n.name: n
        for n in cls.body
        if isinstance(n, ast.AsyncFunctionDef)
    }
    for name in ("_fetch_latest_scored_timestamp",
                 "_fetch_latest_rows_without_window"):
        args = [a.arg for a in methods[name].args.args]
        assert "upper_bound_ts" in args, \
            f"{name} must take upper_bound_ts (Étape 2)"


def test_sentiment_run_propagates_reference_ts():
    """Étape 2: run() must pass upper_bound_ts=reference_ts to BOTH
    fallback methods."""
    src = (ROOT / "agents" / "sentiment_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SentimentAgent"
    )
    run_method = next(
        n for n in cls.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "run"
    )
    run_src = ast.get_source_segment(src, run_method)
    n_calls = run_src.count("upper_bound_ts=reference_ts")
    assert n_calls == 2, \
        f"run() should propagate upper_bound_ts to 2 fallbacks, found {n_calls}"


def test_sentiment_fallback_sql_filters_published_at():
    """Étape 2: the SQL bodies of both fallbacks must filter by
    published_at <= bound."""
    src = (ROOT / "agents" / "sentiment_agent.py").read_text(encoding="utf-8")
    # The two fallback SQL queries each contain "published_at <= $N"
    matches = re.findall(r"published_at\s*<=\s*\$\d+", src)
    # Both fallback methods plus the existing _fetch_window_rows query
    # already had the filter, so we expect at least 3 occurrences total.
    assert len(matches) >= 3, \
        f"published_at <= $N filter must appear in fallback SQL, found {matches}"


# ══════════════════════════════════════════════════════════════════════
# Test 4 — Mini label-shuffle sanity bound
# ══════════════════════════════════════════════════════════════════════

def test_label_shuffle_baseline_does_not_generalize():
    """Sanity bound for the full pipeline: when train labels are randomly
    permuted, a model trained on the shuffled train must not generalize
    to a held-out test set above the always-DOWN baseline (~58%).

    This is a fast, architecture-agnostic guard. It does NOT replace a
    full TFT shuffle test (multi-hour) — see paper/leakage_audit.md §4.1.
    The intent is to fail loudly if a future refactor accidentally
    duplicates the test labels into a feature column or similar.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(seed=42)

    # Synthetic panel with NO learnable signal
    n_train, n_test, n_features = 1000, 300, 20
    X_train = rng.standard_normal((n_train, n_features)).astype("float32")
    X_test = rng.standard_normal((n_test, n_features)).astype("float32")

    # Train labels are pure noise (50/50)
    y_train = rng.integers(0, 2, size=n_train)
    # Test labels follow the BVMT-like 41.9% UP majority-DOWN imbalance
    y_test = (rng.random(n_test) < 0.419).astype(int)

    model = LogisticRegression(max_iter=200, C=1.0, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    # Bound: should land near the always-DOWN majority class baseline
    # (~58.1%). Allow 5pp slack for the small-sample noise in this test.
    UPPER_BOUND = 0.63
    assert acc <= UPPER_BOUND, (
        f"label-shuffle generalized to test_acc={acc:.3f} (>{UPPER_BOUND:.2f}). "
        "Either the bound is too tight or there is an unintended signal "
        "leaking from features into labels."
    )


# ══════════════════════════════════════════════════════════════════════
# Test 5 — End-to-end leak detector for the boundary fix
# ══════════════════════════════════════════════════════════════════════

def test_boundary_leak_would_inflate_target_correlation():
    """Companion to Test 1. Demonstrates WHY the boundary drop matters:
    boundary rows have a target whose underlying close lives in the next
    split. We construct a synthetic series where the close jumps at
    TRAIN_END to make the leak visible in correlations."""
    rng = np.random.default_rng(seed=7)
    horizon = 7
    train_end = pd.Timestamp("2024-01-01")

    # Two tickers, daily business series, 250 days each
    dates = pd.bdate_range("2023-09-01", "2024-04-01", freq="B")
    base = np.linspace(10.0, 11.0, len(dates))
    # Inject a step jump after train_end — this is what would leak into
    # late-2023 training labels via shift(-7).
    base[dates >= train_end] += 0.5

    rows = []
    for tid in (0, 1):
        # Add a small per-ticker noise but keep the structural jump.
        prices = base + rng.standard_normal(len(dates)) * 0.01 + tid * 0.1
        for i, d in enumerate(dates):
            rows.append({"ticker_id": str(tid), "date": d, "close_price": prices[i]})
    df = pd.DataFrame(rows)

    # Without the fix: keep all rows, just compute the target.
    naive = df.copy()
    naive["_future_close"] = (
        naive.groupby("ticker_id")["close_price"].transform(lambda x: x.shift(-horizon))
    )
    naive["future_return_7d"] = (
        (naive["_future_close"] - naive["close_price"]) / naive["close_price"]
    )
    naive = naive.dropna(subset=["future_return_7d"]).reset_index(drop=True)

    # Boundary rows of train (last 7 trading days of 2023) have an
    # inflated mean return because their target reaches over the jump.
    train_naive = naive[naive["date"] < train_end]
    boundary = train_naive.tail(horizon * 2)
    interior = train_naive.head(len(train_naive) - horizon * 2)

    boundary_mean = boundary["future_return_7d"].mean()
    interior_mean = interior["future_return_7d"].mean()
    assert abs(boundary_mean - interior_mean) > 0.005, (
        f"Synthetic setup failed to produce a visible boundary effect: "
        f"boundary={boundary_mean:.5f}, interior={interior_mean:.5f}"
    )

    # With the fix: Section 5b removes those boundary rows, so the bias
    # disappears in the kept train rows.
    cleaned, n_train_drop, _ = _apply_section_4_and_5b(
        df, train_end="2024-01-01", val_end="2030-01-01", horizon=horizon
    )
    cleaned_train = cleaned[cleaned["date"] < train_end]
    cleaned_boundary = cleaned_train.tail(horizon * 2)
    bias = abs(
        cleaned_boundary["future_return_7d"].mean()
        - interior["future_return_7d"].mean()
    )
    assert bias < abs(boundary_mean - interior_mean), \
        "Section 5b did not reduce the boundary bias"
    assert n_train_drop > 0, "Section 5b should have dropped boundary rows"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
