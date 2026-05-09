# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  agents/fundamental_agent.py                                             ║
# ║  BVMT — FundamentalAgent                                                ║
# ║                                                                          ║
# ║  Queries financial_ratios for the latest record per ticker,             ║
# ║  builds XGBoost feature vector, returns FundamentalSignal.             ║
# ║                                                                          ║
# ║  HOW TO TEST:                                                            ║
# ║  python agents/fundamental_agent.py AMEN BANK                          ║
# ║  python agents/fundamental_agent.py SFBT                               ║
# ║  python agents/fundamental_agent.py UNKNOWN                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import asyncio
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

MODEL_PATH   = Path("models/fundamental_xgb.pkl")
ENCODER_PATH = Path("models/fundamental_label_encoder.pkl")
FEATURE_PATH = Path("models/fundamental_features.json")

HEALTH_LABELS = ["CRITICAL", "WEAK", "MODERATE", "STRONG"]


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — FundamentalSignal DATACLASS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class FundamentalSignal:
    ticker:        str
    health_score:  int            # 0=CRITICAL, 1=WEAK, 2=MODERATE, 3=STRONG
    weighted_health_score: float  # continuous score in [0, 3] from class probabilities
    normalized_health_score: float  # weighted_health_score / 3 in [0, 1]
    health_label:  str
    confidence:    float          # XGBoost max class probability (0-1)
    probabilities: dict           # {"CRITICAL": 0.05, "WEAK": 0.1, ...}
    key_ratios:    dict           # actual ratio values for display
    period:        str            # e.g. "FY 2024"
    company_type:  str
    data_source:   str            # query stage that produced the record
    matched_ticker: str           # canonical ticker from DB row
    matched_isin_code: str        # canonical isin from DB row
    error:         Optional[str] = None

    def is_valid(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "ticker":        self.ticker,
            "health_score":  self.health_score,
            "weighted_health_score": round(self.weighted_health_score, 4),
            "normalized_health_score": round(self.normalized_health_score, 4),
            "health_label":  self.health_label,
            "confidence":    round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "key_ratios":    self.key_ratios,
            "period":        self.period,
            "company_type":  self.company_type,
            "data_source":   self.data_source,
            "matched_ticker": self.matched_ticker,
            "matched_isin_code": self.matched_isin_code,
            "error":         self.error,
        }


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATABASE QUERY
#
# Reads from financial_ratios (not financial_statements).
# Gets the most recent FY record with confidence >= 0.5.
# ══════════════════════════════════════════════════════════════════════════

def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def _query_latest_ratios(
    conn,
    key: str,
    period_pattern: Optional[str],
    min_conf: float,
    require_size_proxy: bool = True,
) -> Optional[pd.Series]:
    """Query financial_ratios with tolerant ticker/isin matching and fallback knobs."""
    df = pd.read_sql(
        """
        SELECT
            ticker, isin_code, period, period_end_date, company_type,
            roe, roa, net_margin, nib_margin,
            cost_income, cost_assets,
            equity_ratio, debt_ratio, loan_to_deposit,
            log_total_assets, asset_growth, revenue_growth,
            extraction_confidence,
            CASE
                WHEN ticker = %(key)s OR isin_code = %(key)s THEN 3
                WHEN UPPER(TRIM(ticker)) = UPPER(TRIM(%(key)s)) THEN 2
                WHEN REPLACE(UPPER(ticker), ' ', '') = REPLACE(UPPER(%(key)s), ' ', '') THEN 1
                ELSE 0
            END AS match_rank
        FROM financial_ratios
        WHERE (
            ticker = %(key)s
            OR isin_code = %(key)s
            OR UPPER(TRIM(ticker)) = UPPER(TRIM(%(key)s))
            OR REPLACE(UPPER(ticker), ' ', '') = REPLACE(UPPER(%(key)s), ' ', '')
        )
          AND (%(period_pattern)s IS NULL OR period LIKE %(period_pattern)s)
          AND extraction_confidence >= %(min_conf)s
                    AND (%(require_size_proxy)s = FALSE OR log_total_assets IS NOT NULL)
        ORDER BY match_rank DESC, period_end_date DESC
        LIMIT 1
        """,
        conn,
        params={
            "key": key,
            "period_pattern": period_pattern,
            "min_conf": min_conf,
            "require_size_proxy": require_size_proxy,
        },
    )
    return None if df.empty else df.iloc[0]


def fetch_latest_ratios(ticker: str) -> tuple[Optional[pd.Series], str]:
    """
    Fetch latest financial_ratios record with staged fallback.

    Returns:
      (row, source_stage)
      - row is None when nothing is found
      - source_stage explains which query path succeeded/failed
    """
    conn = get_conn()
    try:
        stages: list[tuple[str, Optional[str], float, bool]] = [
            ("strict_fy_conf_0.50", "FY%", 0.50, True),
            ("fallback_any_period_conf_0.50", None, 0.50, True),
            ("fallback_any_period_conf_0.30", None, 0.30, True),
            ("fallback_any_period_conf_0.30_no_size_proxy", None, 0.30, False),
        ]

        for stage_name, period_pattern, min_conf, require_size_proxy in stages:
            row = _query_latest_ratios(
                conn,
                ticker,
                period_pattern,
                min_conf,
                require_size_proxy=require_size_proxy,
            )
            if row is not None:
                return row, stage_name

        return None, "no_financial_data_after_fallbacks"
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE VECTOR BUILDER
#
# Builds the feature vector in EXACTLY the same order as FEATURE_COLS
# in train_fundamental.py. Order mismatch = wrong predictions.
#
# SIMPLE EXAMPLE:
#   DB row: roe=13.5, roa=1.2, cost_income=38.1, nib_margin=None, ...
#   Vector: [13.5, 1.2, 38.1, ..., 0.0, 0, 0.0, 0, 0.0, 0, 0, 2024]
#                                        ↑nib   ↑flag              ↑year
# ══════════════════════════════════════════════════════════════════════════

def build_feature_vector(row: pd.Series, feature_cols: list) -> np.ndarray:
    """
    Convert a financial_ratios row into an XGBoost feature vector.
    Returns shape (1, n_features).
    """
    type_map = {"bank": 0, "insurance": 1, "leasing": 2, "non_bank": 3}

    def safe(col) -> float:
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0
        return float(v)

    def is_available(col) -> int:
        v = row.get(col)
        return 0 if (v is None or (isinstance(v, float) and np.isnan(v))) else 1

    try:
        year = int(pd.to_datetime(row.get("period_end_date")).year)
    except Exception:
        year = 2020

    feat_dict = {
        # Universal
        "roe":              safe("roe"),
        "roa":              safe("roa"),
        "cost_income":      safe("cost_income"),
        "cost_assets":      safe("cost_assets"),
        "equity_ratio":     safe("equity_ratio"),
        "debt_ratio":       safe("debt_ratio"),
        "log_total_assets": safe("log_total_assets"),
        "asset_growth":     safe("asset_growth"),
        "revenue_growth":   safe("revenue_growth"),
        # Partial coverage
        "net_margin":       safe("net_margin"),
        "has_net_margin":   is_available("net_margin"),
        # Bank-only
        "nib_margin":       safe("nib_margin"),
        "has_nib_margin":   is_available("nib_margin"),
        "loan_to_deposit":  safe("loan_to_deposit"),
        "has_loan_to_deposit": is_available("loan_to_deposit"),
        # Always present
        "company_type_enc": type_map.get(
            str(row.get("company_type", "non_bank")).lower().strip(), 3
        ),
        "year": year,
    }

    # Build in exact column order from feature_cols
    vec = np.array(
        [feat_dict[col] for col in feature_cols],
        dtype=np.float32,
    ).reshape(1, -1)

    return vec


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — FundamentalAgent CLASS
#
# Design patterns (same as TechnicalAgent):
#   Lazy loading:         model loads on first run(), not at import
#   Async:                run() is awaitable for parallel orchestration
#   Graceful degradation: errors return valid signal with error field set
# ══════════════════════════════════════════════════════════════════════════

class FundamentalAgent:
    """
    Classifies company financial health using XGBoost on financial_ratios.
    Called by OrchestratorAgent alongside TechnicalAgent.
    """

    def __init__(self):
        self._model        = None
        self._encoder      = None
        self._feature_cols = None
        self._loaded       = False
        log.info("FundamentalAgent initialized (lazy loading)")

    def _load_model(self):
        if self._loaded:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}\n"
                "Run: python training/train_fundamental.py"
            )
        with open(MODEL_PATH,   "rb") as f: self._model   = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f: self._encoder = pickle.load(f)
        with open(FEATURE_PATH)       as f: self._feature_cols = json.load(f)
        self._loaded = True
        log.info(f"FundamentalAgent model loaded ({len(self._feature_cols)} features)")

    def _make_error_signal(
        self,
        ticker,
        error_msg,
        period="N/A",
        ctype="unknown",
        data_source="none",
        matched_ticker="",
        matched_isin_code="",
    ):
        return FundamentalSignal(
            ticker=ticker, health_score=1, health_label="WEAK",
            weighted_health_score=1.0, normalized_health_score=(1.0 / 3.0),
            confidence=0.0, probabilities={}, key_ratios={},
            period=period, company_type=ctype,
            data_source=data_source,
            matched_ticker=matched_ticker,
            matched_isin_code=matched_isin_code,
            error=error_msg,
        )

    async def run(self, ticker: str) -> FundamentalSignal:
        """
        Main entry point called by OrchestratorAgent.

        Steps:
          1. Load XGBoost model (first call only)
          2. Query most recent FY record from financial_ratios
          3. Build feature vector (17 features)
          4. XGBoost predict_proba → health class + confidence
          5. Return FundamentalSignal

        Returns FundamentalSignal with error set if anything fails.
        OrchestratorAgent checks signal.is_valid() before using it.
        """
        log.info(f"FundamentalAgent.run() → {ticker}")
        t0 = time.perf_counter()

        # Step 1: Load model
        try:
            self._load_model()
        except Exception as e:
            return self._make_error_signal(ticker, str(e))

        # Step 2: Query financial_ratios
        try:
            row, source_stage = fetch_latest_ratios(ticker)
        except Exception as e:
            return self._make_error_signal(ticker, f"DB error: {e}")

        if row is None:
            return self._make_error_signal(
                ticker,
                source_stage,
                data_source=source_stage,
            )

        period = str(row.get("period", "N/A"))
        ctype  = str(row.get("company_type", "unknown"))
        matched_ticker = str(row.get("ticker", ""))
        matched_isin = str(row.get("isin_code", ""))

        # Step 3: Build feature vector
        try:
            X = build_feature_vector(row, self._feature_cols)
        except Exception as e:
            return self._make_error_signal(
                ticker,
                f"feature error: {e}",
                period,
                ctype,
                source_stage,
                matched_ticker,
                matched_isin,
            )

        # Step 4: XGBoost predict
        try:
            proba      = self._model.predict_proba(X)[0]
            pred_class = int(np.argmax(proba))
            confidence = float(np.max(proba))
            weighted_health_score = float(np.dot(proba, np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)))
            normalized_health_score = weighted_health_score / 3.0
            probabilities = {
                label: round(float(p), 4)
                for label, p in zip(HEALTH_LABELS, proba)
            }
        except Exception as e:
            return self._make_error_signal(
                ticker,
                f"predict error: {e}",
                period,
                ctype,
                source_stage,
                matched_ticker,
                matched_isin,
            )

        # Step 5: Build key_ratios dict for display
        key_ratios = {}
        display_map = [
            ("roe",             "ROE (%)"),
            ("roa",             "ROA (%)"),
            ("cost_income",     "Cost-income (%)"),
            ("cost_assets",     "Cost/assets (%)"),
            ("equity_ratio",    "Equity ratio (%)"),
            ("debt_ratio",      "Debt ratio (%)"),
            ("net_margin",      "Net margin (%)"),
            ("nib_margin",      "NIB margin (%)"),
            ("loan_to_deposit", "Loan/deposit (%)"),
            ("log_total_assets","Log total assets"),
            ("asset_growth",    "Asset growth (%)"),
            ("revenue_growth",  "Revenue growth (%)"),
        ]
        for col, label in display_map:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                key_ratios[label] = round(float(val), 2)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        health_label = HEALTH_LABELS[pred_class]

        log.info(
            f"FundamentalAgent: {ticker} → {health_label} "
            f"({confidence*100:.0f}% confidence) [{elapsed_ms:.1f}ms]"
        )

        return FundamentalSignal(
            ticker        = ticker,
            health_score  = pred_class,
            weighted_health_score = weighted_health_score,
            normalized_health_score = normalized_health_score,
            health_label  = health_label,
            confidence    = confidence,
            probabilities = probabilities,
            key_ratios    = key_ratios,
            period        = period,
            company_type  = ctype,
            data_source   = source_stage,
            matched_ticker = matched_ticker,
            matched_isin_code = matched_isin,
            error         = None,
        )


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — QUICK TEST
# ══════════════════════════════════════════════════════════════════════════

async def _test(ticker: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    print(f"\nTesting FundamentalAgent on: {ticker}")
    print("=" * 52)

    agent  = FundamentalAgent()
    signal = await agent.run(ticker)

    print(f"Model    : {MODEL_PATH}  (exists={MODEL_PATH.exists()})")
    print()

    if not signal.is_valid():
        print(f"ERROR: {signal.error}")
        if signal.error == "no_financial_data":
            print("  → This ticker has no records in financial_ratios")
            print("  → Run training/train_fundamental.py to check coverage")
        elif "not found" in str(signal.error):
            print("  → Run: python training/train_fundamental.py")
        return

    print(f"{ticker} → {signal.health_label} ({signal.confidence*100:.0f}%)")
    print()
    print(f"  Health Score : {signal.health_score}/3  ({signal.health_label})")
    print(f"  Confidence   : {signal.confidence*100:.1f}%")
    print(f"  Period       : {signal.period}")
    print(f"  Company type : {signal.company_type}")
    print()
    print("  Class probabilities:")
    for label, prob in signal.probabilities.items():
        bar = "█" * int(prob * 30)
        print(f"    {label:<10}: {prob*100:5.1f}%  {bar}")
    print()
    print("  Key ratios (latest FY):")
    for ratio, val in signal.key_ratios.items():
        print(f"    {ratio:<28}: {val}")
    print()
    print("  signal.to_dict():")
    print(json.dumps(signal.to_dict(), indent=2))


if __name__ == "__main__":
    ticker = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AMEN BANK"
    asyncio.run(_test(ticker))
