"""
/prediction/{ticker}, /cache/* endpoints.

This is the core endpoint.  It runs the full OrchestratorAgent pipeline
and caches results for 30 minutes.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

# ── make project root importable ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.orchestrator import OrchestratorAgent
from app.backend.cache import TickerCache
from app.backend.models import (
    AIArticleSection,
    CacheStatusResponse,
    DataQualitySection,
    FundamentalSection,
    GraphSection,
    NeighborInfo,
    OrchestrationSection,
    PredictionMetadata,
    PredictionResponse,
    PriceHorizonPoint,
    SentimentSection,
    TechnicalSection,
    TopFeature,
    XAISection,
    RecentArticle,
)

router = APIRouter(tags=["prediction"])

# ── Shared cache instance (imported by other routers) ────────────────
import os

# Cache TTL (seconds) configurable via env var; default 2 hours (7200s)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 7200))
cache = TickerCache(ttl_seconds=CACHE_TTL_SECONDS)

# ── Shared orchestrator instance (lazy model loading inside) ─────────
orchestrator = OrchestratorAgent()


# ═══════════════════════════════════════════════════════════════════
# State → Response serializers
# ═══════════════════════════════════════════════════════════════════

def _company_type_from_state(data: Dict[str, Any]) -> str:
    tech = data.get("agent_outputs", {}).get("technical", {})
    fund = data.get("agent_outputs", {}).get("fundamental", {})
    ct = fund.get("company_type", "")
    if ct:
        return ct
    ticker = data.get("ticker", "")
    banks = {
        "AMEN BANK", "BIAT", "BNA", "STB", "BH BANK",
        "ATTIJARI BANK", "ATB", "UBCI", "UIB", "BT", "WIFACK INT BANK",
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


def _build_metadata(data: Dict[str, Any], cache_hit: bool, computation_seconds: float) -> PredictionMetadata:
    return PredictionMetadata(
        ticker=data.get("ticker", ""),
        isin_code=data.get("isin_code") or None,
        company_type=_company_type_from_state(data),
        run_timestamp=data.get("created_at") or datetime.now().isoformat(),
        cache_hit=cache_hit,
        computation_time_seconds=round(computation_seconds, 2),
    )


def _build_data_quality(data: Dict[str, Any]) -> DataQualitySection:
    skip_flags: List[str] = []
    if data.get("skip_sentiment"):
        skip_flags.append("sentiment")
    if data.get("skip_fundamental"):
        skip_flags.append("fundamental")

    da = data.get("agent_outputs", {}).get("data", {})
    return DataQualitySection(
        rows_available=data.get("price_rows_available", 0),
        latest_date=None,
        null_count=0,
        news_article_count=data.get("news_articles_count", 0),
        financial_statements_available=bool(data.get("has_recent_financials", False)),
        data_score=round(float(data.get("data_quality_score", 0.0)), 4),
        years_of_data=data.get("years_of_price_data", 0),
        skip_flags=skip_flags,
    )


def _build_technical(data: Dict[str, Any]) -> TechnicalSection:
    tech = data.get("agent_outputs", {}).get("technical", {})
    if not tech:
        return TechnicalSection(error="Technical agent produced no output")

    dq = tech.get("data_quality", {})
    pred_prices = dq.get("predicted_price_horizons", [])
    quantiles = tech.get("quantiles", {})

    # Build 7-day price horizon
    horizon: List[PriceHorizonPoint] = []
    if pred_prices:
        # Price-mode: actual prices
        for i, price in enumerate(pred_prices):
            horizon.append(PriceHorizonPoint(
                day_number=i + 1,
                predicted_price=round(float(price), 3),
                predicted_return_pct=None,
            ))
    else:
        # Quantile-mode: use q50 as single-value proxy
        q50 = float(quantiles.get("0.5", 0.0))
        for i in range(7):
            horizon.append(PriceHorizonPoint(
                day_number=i + 1,
                predicted_price=round(q50, 4),
                predicted_return_pct=round(q50 * 100, 2),
            ))

    # Build top features
    raw_features = tech.get("top_features", [])
    top_feats: List[TopFeature] = []
    for f in raw_features:
        top_feats.append(TopFeature(
            feature_name=f.get("feature", ""),
            current_value=float(f.get("current", 0.0)),
            z_score=float(f.get("z_score", 0.0)),
            interpretation=f.get("note", ""),
        ))

    encoder_len = dq.get("max_encoder_length", 60)
    pred_len = dq.get("max_prediction_length", 7)

    return TechnicalSection(
        direction=tech.get("direction", "UNKNOWN"),
        confidence=round(float(tech.get("confidence", 0.0)), 4),
        signal_str=tech.get("signal_str", ""),
        price_horizon=horizon,
        top_features=top_feats,
        model_info={
            "encoder_window_days": encoder_len,
            "prediction_horizon_days": pred_len,
            "training_period": "2016–2023",
            "val_period": "2024",
            "test_period": "2025",
            "model_accuracy_pct": 77.4,
            "checkpoint": dq.get("model_checkpoint", ""),
            "prediction_mode": dq.get("prediction_mode", ""),
        },
        error=tech.get("error"),
        expected_move_pct=dq.get("expected_move_pct"),
        current_close=dq.get("current_close"),
    )


def _build_fundamental(data: Dict[str, Any]) -> FundamentalSection:
    fund = data.get("agent_outputs", {}).get("fundamental", {})
    if not fund:
        return FundamentalSection(error="Fundamental agent produced no output")

    return FundamentalSection(
        health_class=fund.get("health_label", "UNKNOWN"),
        confidence=round(float(fund.get("confidence", 0.0)), 4),
        health_score=round(float(fund.get("weighted_health_score", 0.0)), 4),
        normalized_health_score=round(float(fund.get("normalized_health_score", 0.0)), 4),
        ratios=fund.get("key_ratios", {}),
        period=fund.get("period"),
        company_type=fund.get("company_type"),
        model_info={
            "algorithm": "XGBoost",
            "cross_validation": "Leave-One-Ticker-Out",
            "health_classes": ["CRITICAL", "WEAK", "MODERATE", "STRONG"],
            "features_used": 17,
            "data_source": fund.get("data_source", ""),
        },
        error=fund.get("error"),
    )


def _build_sentiment(data: Dict[str, Any]) -> SentimentSection:
    sent = data.get("agent_outputs", {}).get("sentiment", {})
    if not sent:
        return SentimentSection()

    score = float(sent.get("sentiment_signal", 0.0))
    label = (sent.get("dominant_label") or "neutral").upper()
    conf = float(sent.get("confidence", 0.0))
    article_count = int(sent.get("article_count", 0))

    return SentimentSection(
        score=round(score, 4),
        label=label,
        confidence=round(conf, 4),
        article_count=article_count,
        date_range={
            "anchor": sent.get("anchor_timestamp"),
            "window_days": sent.get("window_days", 30),
        },
        recent_articles=[],
        model_info={
            "model_name": "bardsai/finance-sentiment-fr-base",
            "language": "French",
            "source": "ilboursa.com",
            "scoring": "offline pre-computed",
        },
    )


def _build_graph(data: Dict[str, Any]) -> GraphSection:
    graph_out = data.get("agent_outputs", {}).get("graph", {})
    if not graph_out:
        return GraphSection()

    # Direction / signal
    g_dir = graph_out.get("graph_direction", "FLAT")
    signal_map = {"UP": "PULL_UP", "DOWN": "PUSH_DOWN", "FLAT": "NEUTRAL"}
    signal = signal_map.get(g_dir, "NEUTRAL")

    # Top neighbors
    neighbors: List[NeighborInfo] = []
    for n in (graph_out.get("related_stocks_top10") or [])[:10]:
        corr = float(n.get("correlation", 0.0))
        inf_type = "leading" if corr > 0.6 else "concurrent"
        neighbors.append(NeighborInfo(
            ticker=n.get("ticker", ""),
            correlation=round(corr, 4),
            direction=None,
            influence_type=inf_type,
            explanation=f"Correlation: {corr:.2f}",
            relation_score=round(float(n.get("relation_score", 0.0)), 4),
            sector=n.get("sector"),
        ))

    # Graph stats
    gate = graph_out.get("gate", {})

    return GraphSection(
        signal=signal,
        confidence=round(float(graph_out.get("confidence", 0.0)), 4),
        neighborhood_score=round(float(graph_out.get("graph_score", 0.0)), 4),
        top_neighbors=neighbors,
        graph_stats={
            "node_index": graph_out.get("node_index"),
            "class_priors": graph_out.get("class_priors", {}),
            "logit_adjust_tau": graph_out.get("logit_adjust_tau"),
        },
        confidence_gating_applied=not bool(graph_out.get("used_in_ensemble", True)),
        class_probs=graph_out.get("graph_class_probs", {}),
    )


def _build_xai(data: Dict[str, Any]) -> XAISection:
    xai = data.get("agent_outputs", {}).get("xai", {})
    if not xai:
        return XAISection()

    tech_xai = xai.get("technical") or {}
    fund_xai = xai.get("fundamental") or {}
    graph_xai = xai.get("graph") or {}

    return XAISection(
        technical_explanation=tech_xai,
        fundamental_explanation=fund_xai,
        graph_explanation=graph_xai,
    )


def _build_orchestration(data: Dict[str, Any]) -> OrchestrationSection:
    weights = data.get("weights", {})
    direction = data.get("final_direction", "NEUTRAL")
    confidence = float(data.get("overall_confidence", 0.0))
    final_score = round(confidence * 100, 1)

    # Build signal string
    label = "BULLISH" if direction == "UP" else ("BEARISH" if direction == "DOWN" else "NEUTRAL")
    signal_str = f"{label} {int(confidence * 100)}%"

    # Agents that were skipped
    failed: List[str] = []
    if data.get("skip_sentiment"):
        failed.append("sentiment (skipped: no news)")
    if data.get("skip_fundamental"):
        failed.append("fundamental (skipped: no recent financials)")

    graph_out = data.get("agent_outputs", {}).get("graph", {})
    if graph_out and not graph_out.get("used_in_ensemble", True):
        failed.append("graph (gated: low confidence)")

    # Decision explanation
    direction_word = "rise" if direction == "UP" else ("fall" if direction == "DOWN" else "remain stable")
    explanation = (
        f"The BVMT AI system predicts {data.get('ticker', 'this stock')} will {direction_word} "
        f"over the next 7 days with {int(confidence * 100)}% composite confidence. "
        f"The TFT model contributed {int(weights.get('technical', 0) * 100)}% weight, "
        f"fundamental health contributed {int(weights.get('fundamental', 0) * 100)}%, "
        f"news sentiment contributed {int(weights.get('sentiment', 0) * 100)}%, "
        f"and graph relationships contributed {int(weights.get('graph', 0) * 100)}%."
    )

    return OrchestrationSection(
        final_direction=direction,
        final_confidence=round(confidence, 4),
        final_score=final_score,
        signal_str=signal_str,
        agent_weights=weights,
        agent_contributions={
            "technical": data.get("agent_outputs", {}).get("technical", {}).get("confidence", 0.0),
            "fundamental": data.get("agent_outputs", {}).get("fundamental", {}).get("confidence", 0.0),
            "sentiment": data.get("agent_outputs", {}).get("sentiment", {}).get("confidence", 0.0),
            "graph": data.get("agent_outputs", {}).get("graph", {}).get("confidence", 0.0),
        },
        agents_failed=failed,
        decision_explanation=explanation,
    )


def _build_explanation_facts(data: Dict[str, Any]) -> Dict[str, Any]:
    """Build deterministic explanation facts from agent outputs. No LLM."""
    orch = data.get("final_direction", "NEUTRAL")
    conf = round(float(data.get("overall_confidence", 0.0)) * 100)

    tech = data.get("agent_outputs", {}).get("technical", {})
    fund = data.get("agent_outputs", {}).get("fundamental", {})
    sent = data.get("agent_outputs", {}).get("sentiment", {})
    graph_out = data.get("agent_outputs", {}).get("graph", {})

    tech_reasons = []
    if tech.get("top_features"):
        for f in tech["top_features"][:3]:
            note = f.get("note", "")
            if note:
                tech_reasons.append(note)
    if not tech_reasons:
        tech_reasons.append(tech.get("signal_str", "TFT prediction"))

    fund_reasons = []
    health = fund.get("health_label", "")
    if health:
        fund_reasons.append(f"Health: {health}")
    for k in ["roe", "debt_to_equity", "current_ratio"]:
        v = fund.get("key_ratios", {}).get(k)
        if v is not None:
            fund_reasons.append(f"{k}: {v}")

    sent_label = (sent.get("dominant_label") or "neutral").upper()
    sent_reasons = []
    if sent.get("article_count", 0) > 0:
        sent_reasons.append(f"{sent['article_count']} articles analyzed")
        sent_reasons.append(f"Signal: {float(sent.get('sentiment_signal', 0)):+.3f}")
    else:
        sent_reasons.append("No news available")

    graph_reasons = []
    g_dir = graph_out.get("graph_direction", "FLAT")
    if graph_out.get("related_stocks_top10"):
        top3 = [n.get("ticker", "") for n in graph_out["related_stocks_top10"][:3]]
        graph_reasons.append(f"Top peers: {', '.join(top3)}")
    graph_reasons.append(f"Peer consensus: {g_dir}")

    return {
        "verdict": orch,
        "confidence": conf,
        "technical": {
            "summary": tech.get("signal_str", "TFT model output"),
            "reasons": tech_reasons,
        },
        "fundamental": {
            "summary": f"Health class: {health or 'N/A'}",
            "reasons": fund_reasons,
        },
        "sentiment": {
            "summary": f"Sentiment: {sent_label}",
            "reasons": sent_reasons,
        },
        "graph": {
            "summary": f"Peers: {g_dir}",
            "reasons": graph_reasons,
        },
        "explain_version": "v1",
    }


def _serialize_state(
    data: Dict[str, Any],
    cache_hit: bool,
    computation_seconds: float,
) -> PredictionResponse:
    return PredictionResponse(
        metadata=_build_metadata(data, cache_hit, computation_seconds),
        data_quality=_build_data_quality(data),
        technical=_build_technical(data),
        fundamental=_build_fundamental(data),
        sentiment=_build_sentiment(data),
        graph=_build_graph(data),
        xai=_build_xai(data),
        orchestration=_build_orchestration(data),
        ai_article=AIArticleSection(content=None),
        explanation_facts=_build_explanation_facts(data),
        execution_log=data.get("execution_log", []),
    )


# ═══════════════════════════════════════════════════════════════════
# Background precompute
# ═══════════════════════════════════════════════════════════════════

async def _precompute_all():
    """Background task: run all tickers sequentially, log progress."""
    import asyncpg, os
    from dotenv import load_dotenv
    load_dotenv()

    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "bvmt_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )
    try:
        rows = await conn.fetch("""
            SELECT DISTINCT ticker FROM daily_prices
            WHERE ticker IS NOT NULL AND TRIM(ticker) <> ''
            ORDER BY ticker
        """)
    finally:
        await conn.close()

    tickers = [r["ticker"] for r in rows]
    total = len(tickers)
    print(f"[Precompute] Starting background computation for {total} tickers …")

    for i, ticker in enumerate(tickers, 1):
        try:
            t0 = time.time()
            state = await orchestrator.run(ticker)
            elapsed = time.time() - t0
            cache.set(ticker, state.to_dict(), computation_seconds=elapsed)
            print(f"[Precompute] [{i}/{total}] {ticker} done in {elapsed:.1f}s")
        except Exception as exc:
            print(f"[Precompute] [{i}/{total}] {ticker} FAILED: {exc}")

    print("[Precompute] All tickers processed.")


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════

@router.get("/prediction/{ticker}", response_model=PredictionResponse)
async def get_prediction(ticker: str):
    """Run or return cached full agent pipeline for one ticker."""
    ticker = ticker.strip().upper()

    entry = cache.get(ticker)
    if entry is not None:
        return _serialize_state(entry.data, cache_hit=True, computation_seconds=entry.computation_seconds)

    # Cache miss — run agents
    try:
        t0 = time.time()
        state = await orchestrator.run(ticker)
        elapsed = time.time() - t0
        cache.set(ticker, state.to_dict(), computation_seconds=elapsed)
        return _serialize_state(state.to_dict(), cache_hit=False, computation_seconds=elapsed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {exc}")


@router.get("/cache/status", response_model=CacheStatusResponse)
async def cache_status():
    """Return how many tickers are cached and when each was last computed."""
    s = cache.status()
    return CacheStatusResponse(
        cached_count=s["cached_count"],
        ttl_seconds=s["ttl_seconds"],
        entries=s["entries"],
    )


@router.post("/cache/refresh/{ticker}", response_model=PredictionResponse)
async def refresh_ticker(ticker: str):
    """Force a fresh agent run for one ticker, ignoring cache."""
    ticker = ticker.strip().upper()
    cache.clear(ticker)

    try:
        t0 = time.time()
        state = await orchestrator.run(ticker)
        elapsed = time.time() - t0
        cache.set(ticker, state.to_dict(), computation_seconds=elapsed)
        return _serialize_state(state.to_dict(), cache_hit=False, computation_seconds=elapsed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {exc}")


@router.post("/cache/precompute", status_code=202)
async def precompute(background_tasks: BackgroundTasks):
    """Trigger background computation for all tickers. Returns 202 immediately."""
    background_tasks.add_task(_precompute_all)
    return {"message": "Background precompute started. Poll /cache/status for progress."}
