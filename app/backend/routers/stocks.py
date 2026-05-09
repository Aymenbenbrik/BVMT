"""
/stocks, /market-overview, /stock-history/{ticker} endpoints.

These read directly from PostgreSQL — no agent pipeline needed.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

from app.backend.models import (
    MarketOverviewResponse,
    PricePoint,
    StockHistoryResponse,
    StockListItem,
    StockListResponse,
)

load_dotenv()

router = APIRouter(tags=["stocks"])

# ── DB helper ────────────────────────────────────────────────────────

async def _get_conn() -> asyncpg.Connection:
    return await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "bvmt_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )


def _get_company_type(ticker: str) -> str:
    """Classify ticker into bank / insurance / non_bank."""
    banks = {
        "AMEN BANK", "BIAT", "BNA", "STB", "BH BANK",
        "ATTIJARI BANK", "ATB", "UBCI", "UIB", "BT",
        "WIFACK INT BANK",
    }
    insurance = {
        "STAR", "ASTREE", "BH ASSURANCE", "ASSUR MAGHREBIA",
        "ASSU MAGHREBIA VIE", "TUNIS RE", "BNA ASSURANCES",
    }
    if ticker in banks:
        return "bank"
    if ticker in insurance:
        return "insurance"
    return "non_bank"


# ── helpers for cache access (imported at call time to avoid circular) ─

def _cache():
    """Lazy import to avoid circular dependency with prediction router."""
    from app.backend.routers.prediction import cache
    return cache


# ── GET /stocks ──────────────────────────────────────────────────────

@router.get("/stocks", response_model=StockListResponse)
async def list_stocks():
    """Return all unique tickers from daily_prices with metadata."""
    conn = await _get_conn()
    try:
        rows = await conn.fetch("""
            SELECT
                dp.ticker,
                dp.isin_code,
                MIN(dp.seance) AS first_seen,
                MAX(dp.seance) AS last_seen,
                COUNT(*) AS rows_count
            FROM daily_prices dp
            WHERE dp.ticker IS NOT NULL
              AND TRIM(dp.ticker) <> ''
            GROUP BY dp.ticker, dp.isin_code
            ORDER BY dp.ticker
        """)
    finally:
        await conn.close()

    c = _cache()
    items: List[StockListItem] = []
    removed: List[str] = []
    total_before = len(rows)

    for row in rows:
        ticker = row["ticker"]
        rows_count = int(row.get("rows_count", 0))
        # Filter: exclude tickers with low coverage (< 500 rows)
        if rows_count < 500:
            removed.append(ticker)
            continue
        entry = c.get(ticker)
        cached = entry is not None

        direction = None
        confidence = None
        final_score = None
        expected_move_pct = None
        current_close = None

        if cached:
            d = entry.data
            direction = d.get("final_direction")
            confidence = d.get("overall_confidence")
            # Compute a 0-100 score from confidence
            if confidence is not None:
                final_score = round(confidence * 100, 1)
                
            tech = d.get("agent_outputs", {}).get("technical", {})
            dq = tech.get("data_quality", {})
            expected_move_pct = dq.get("expected_move_pct")
            current_close = dq.get("current_close")

        items.append(StockListItem(
            ticker=ticker,
            isin_code=row["isin_code"],
            company_type=_get_company_type(ticker),
            sector=None,
            first_seen=str(row["first_seen"]) if row["first_seen"] else None,
            last_seen=str(row["last_seen"]) if row["last_seen"] else None,
            cached=cached,
            direction=direction,
            confidence=confidence,
            final_score=final_score,
            expected_move_pct=expected_move_pct,
            current_close=current_close,
        ))

    return StockListResponse(stocks=items, total=len(items))


@router.get("/stocks/coverage-report")
async def stocks_coverage_report():
    """Return which tickers were removed for low coverage and how many remain."""
    conn = await _get_conn()
    try:
        rows = await conn.fetch("""
            SELECT dp.ticker, COUNT(*) AS rows_count
            FROM daily_prices dp
            WHERE dp.ticker IS NOT NULL AND TRIM(dp.ticker) <> ''
            GROUP BY dp.ticker
            ORDER BY dp.ticker
        """)
    finally:
        await conn.close()

    total_before = len(rows)
    removed = [r["ticker"] for r in rows if int(r.get("rows_count", 0)) < 500]
    remaining = total_before - len(removed)

    from app.backend.models import CoverageReportResponse

    return CoverageReportResponse(
        total_before=total_before,
        remaining_count=remaining,
        removed_count=len(removed),
        removed_tickers=removed,
    )


# ── GET /market-overview ─────────────────────────────────────────────

@router.get("/market-overview", response_model=MarketOverviewResponse)
async def market_overview():
    """Aggregate all stocks with cached analysis results."""
    stock_resp = await list_stocks()
    stocks = stock_resp.stocks

    bull = sum(1 for s in stocks if s.direction == "UP")
    bear = sum(1 for s in stocks if s.direction == "DOWN")
    hold = sum(1 for s in stocks if s.direction in ("NEUTRAL", "HOLD"))
    uncached = sum(1 for s in stocks if not s.cached)

    confs = [s.confidence for s in stocks if s.confidence is not None]
    avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

    # Determine mood
    if bull == 0 and bear == 0:
        mood = "UNKNOWN"
    elif abs(bull - bear) <= max(1, int(0.05 * (bull + bear))):
        mood = "MIXED"
    elif bull > bear:
        mood = "BULLISH"
    else:
        mood = "BEARISH"

    return MarketOverviewResponse(
        stocks=stocks,
        total=len(stocks),
        bull_count=bull,
        bear_count=bear,
        hold_count=hold,
        uncached_count=uncached,
        average_confidence=avg_conf,
        market_mood=mood,
    )


# ── GET /stock-history/{ticker} ──────────────────────────────────────

@router.get("/stock-history/{ticker}", response_model=StockHistoryResponse)
async def stock_history(ticker: str):
    """Return last 252 trading days of price data + predicted prices if cached."""
    conn = await _get_conn()
    try:
        rows = await conn.fetch("""
            SELECT
                dp.seance   AS date,
                dp.cloture  AS close,
                dp.ouverture AS open,
                dp.plus_haut AS high,
                dp.plus_bas  AS low,
                dp.quantite_negociee AS volume,
                ci.daily_return
            FROM daily_prices dp
            LEFT JOIN computed_indicators ci
                ON dp.isin_code = ci.isin_code AND dp.seance = ci.seance
            WHERE dp.ticker = $1
              AND dp.cloture > 0
            ORDER BY dp.seance DESC
            LIMIT 252
        """, ticker)
    finally:
        await conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

    # Reverse to chronological order
    history = []
    for r in reversed(rows):
        history.append(PricePoint(
            date=str(r["date"]),
            close=float(r["close"]),
            open=float(r["open"]) if r["open"] else None,
            high=float(r["high"]) if r["high"] else None,
            low=float(r["low"]) if r["low"] else None,
            volume=float(r["volume"]) if r["volume"] else None,
            daily_return=float(r["daily_return"]) if r["daily_return"] else None,
            is_predicted=False,
        ))

    # Predicted prices from cache if available (future horizon for all stocks).
    predicted: List[PricePoint] = []
    entry = _cache().get(ticker)
    if entry is not None:
        tech = entry.data.get("agent_outputs", {}).get("technical", {})
        dq = tech.get("data_quality", {})
        pred_prices = dq.get("predicted_price_horizons", [])
        if pred_prices and history:
            from datetime import timedelta
            import datetime as dt

            last_date_str = history[-1].date
            try:
                last_date = dt.date.fromisoformat(last_date_str)
            except ValueError:
                last_date = dt.date.today()

            for i, price in enumerate(pred_prices):
                # Skip weekends naively (add 1 day, skip Sat/Sun)
                day_offset = i + 1
                pred_date = last_date + timedelta(days=day_offset)
                while pred_date.weekday() >= 5:
                    pred_date += timedelta(days=1)

                predicted.append(PricePoint(
                    date=str(pred_date),
                    close=round(float(price), 3),
                    is_predicted=True,
                ))

    return StockHistoryResponse(
        ticker=ticker,
        history=history,
        predicted=predicted,
    )
