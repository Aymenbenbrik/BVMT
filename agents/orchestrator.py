"""
OrchestratorAgent: runs the multi-agent pipeline and assembles final output.

Current implemented agents:
- DataAgent
- TechnicalAgent (TFT price/quantile inference)
- FundamentalAgent (XGBoost financial health)
- SentimentAgent (pre-scored news sentiment)
- GraphAgent (GAT neighborhood propagation)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

from agents.agent_state import AgentState
from agents.data_agent import DataAgent
from agents.fundamental_agent import FundamentalAgent
from agents.graph_agent import GraphAgent
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from agents.xai_agent import XAIAgent


class OrchestratorAgent:
    def __init__(self, db_url: str | None = None):
        self.data_agent = DataAgent()
        self.technical_agent = TechnicalAgent()
        self.fundamental_agent = FundamentalAgent()
        self.graph_agent = GraphAgent()
        self.xai_agent = XAIAgent()
        self.db_url = db_url
        self.sentiment_available = True
        # Graph contribution gate: avoid adding noisy low-conviction signals.
        self.graph_min_confidence = 0.20
        self.graph_min_abs_score = 0.08

    async def run(
        self,
        ticker: str,
        user_role: str = "investor",
        seance: str | None = None,
    ) -> AgentState:
        state = AgentState(
            ticker=ticker,
            request_id=str(uuid.uuid4())[:8],
            user_role=user_role,
            created_at=datetime.now().isoformat(),
            seance=(seance or ""),
        )
        state.log(
            f"Orchestrator starting for {ticker} (id={state.request_id}"
            + (f", seance={seance}" if seance else "")
            + ")"
        )

        state = await self._resolve_identity(state)

        # Step 1: data quality
        state.log("Step 1: Running DataAgent...")
        state = await self.data_agent.run(state)

        # Step 2: SentimentAgent
        if not state.skip_sentiment:
            sentiment_agent = SentimentAgent(db_url=self.db_url)
            state = await sentiment_agent.run(state)
            state.log(
                f"Sentiment signal: "
                f"{state.agent_outputs['sentiment']['sentiment_signal']:+.4f}  "
                f"conf={state.confidence_scores['sentiment']:.3f}"
            )
        else:
            state.agent_outputs['sentiment'] = {
                'sentiment_signal': 0.0,
                'dominant_label': 'neutral',
                'article_count': 0,
                'confidence': 0.0,
            }
            state.confidence_scores['sentiment'] = 0.0
            state.log("SentimentAgent skipped (insufficient articles)")

        # Step 3: dynamic weights
        state.log("Step 3: Adjusting agent weights...")
        state = self._adjust_weights(state)

        # Step 4: run core scalar agents in parallel
        state.log("Step 4: Running TechnicalAgent + FundamentalAgent...")
        # Propagate state.seance to FundamentalAgent so back-tested calls
        # only see ratios that were "as-if-known" on that date. Empty
        # string => live use (FundamentalAgent defaults to today).
        tech_signal, fund_signal = await asyncio.gather(
            self.technical_agent.run(state.ticker),
            self.fundamental_agent.run(
                state.ticker,
                prediction_date=(state.seance or None),
            ),
        )

        state.agent_outputs["technical"] = tech_signal.to_dict()
        state.agent_outputs["fundamental"] = fund_signal.to_dict()

        state.confidence_scores["technical"] = (
            float(tech_signal.confidence) if tech_signal.is_valid() else 0.0
        )
        state.confidence_scores["fundamental"] = (
            float(fund_signal.confidence) if fund_signal.is_valid() else 0.0
        )

        # Step 5: run graph propagation using outputs from previous agents.
        # GraphAgent needs technical/sentiment/fundamental values already in state.
        state.log("Step 5: Running GraphAgent...")
        state = await self.graph_agent.run(state)

        # Step 5.5: Generate explanations using XAIAgent
        state.log("Step 5.5: Running XAIAgent...")
        state = await self.xai_agent.run(state)

        # Step 6: final assembly
        state.log("Step 6: Assembling final result...")
        self._assemble_final_result(state, tech_signal, fund_signal)
        state.log(
            f"Pipeline complete. direction={state.final_direction}, "
            f"confidence={state.overall_confidence:.3f}"
        )
        return state

    async def _resolve_identity(self, state: AgentState) -> AgentState:
        """
        Resolve user input to canonical (ticker, isin_code) from company_metadata.
        Falls back to daily_prices if company_metadata has no match.
        """
        requested = (state.ticker or "").strip()
        if not requested:
            return state

        try:
            rows = await self.data_agent.query_db(
                """
                SELECT ticker, isin_code,
                       CASE
                           WHEN ticker = $1 OR isin_code = $1 THEN 4
                           WHEN UPPER(TRIM(ticker)) = UPPER(TRIM($1)) THEN 3
                           WHEN REPLACE(UPPER(ticker), ' ', '') = REPLACE(UPPER($1), ' ', '') THEN 2
                           ELSE 1
                       END AS match_rank
                FROM company_metadata
                WHERE ticker = $1
                   OR isin_code = $1
                   OR UPPER(TRIM(ticker)) = UPPER(TRIM($1))
                   OR REPLACE(UPPER(ticker), ' ', '') = REPLACE(UPPER($1), ' ', '')
                ORDER BY match_rank DESC, total_trading_days DESC NULLS LAST
                LIMIT 1
                """,
                requested,
            )

            if not rows:
                rows = await self.data_agent.query_db(
                    """
                    SELECT dp.ticker, dp.isin_code,
                           MAX(dp.seance) AS last_seen
                    FROM daily_prices dp
                    WHERE dp.ticker = $1
                       OR dp.isin_code = $1
                       OR UPPER(TRIM(dp.ticker)) = UPPER(TRIM($1))
                       OR REPLACE(UPPER(dp.ticker), ' ', '') = REPLACE(UPPER($1), ' ', '')
                    GROUP BY dp.ticker, dp.isin_code
                    ORDER BY last_seen DESC
                    LIMIT 1
                    """,
                    requested,
                )

            if rows:
                resolved_ticker = str(rows[0].get("ticker") or requested)
                resolved_isin = str(rows[0].get("isin_code") or "")
                state.ticker = resolved_ticker
                state.isin_code = resolved_isin
                state.log(
                    f"Identity resolved: input='{requested}' -> ticker='{resolved_ticker}', isin='{resolved_isin or 'N/A'}'"
                )
            else:
                state.log("Identity resolution: no canonical match found; using input ticker")

        except Exception as e:
            state.log(f"Identity resolution error: {e}")

        return state

    def _adjust_weights(self, state: AgentState) -> AgentState:
        """
        Dynamic weight adjustments based on data quality.
        Then remove sentiment weight if sentiment agent is unavailable.
        """
        w = state.weights.copy()

        if state.news_articles_count == 0:
            state.log("  No news -> heavy sentiment reduction and skip")
            reduction = w["sentiment"] * 0.80
            w["sentiment"] -= reduction
            w["technical"] += reduction * 0.50
            w["fundamental"] += reduction * 0.30
            w["graph"] += reduction * 0.20
            state.skip_sentiment = True
        elif state.news_articles_count < 5:
            state.log(f"  Sparse news ({state.news_articles_count}) -> reducing sentiment weight, keep enabled")
            reduction = w["sentiment"] * 0.55
            w["sentiment"] -= reduction
            w["technical"] += reduction * 0.50
            w["fundamental"] += reduction * 0.30
            w["graph"] += reduction * 0.20

        if not state.has_recent_financials:
            state.log("  No recent financials -> reducing fundamental weight")
            reduction = w["fundamental"] * 0.50
            w["fundamental"] -= reduction
            w["technical"] += reduction * 0.70
            w["graph"] += reduction * 0.30
            state.skip_fundamental = True

        if state.price_rows_available < 252:
            state.log(f"  Low price rows ({state.price_rows_available}) -> reducing technical weight")
            reduction = w["technical"] * 0.40
            w["technical"] -= reduction
            w["fundamental"] += reduction * 0.35
            w["sentiment"] += reduction * 0.35
            w["graph"] += reduction * 0.30

        total = sum(w.values())
        if total <= 0:
            w = {"technical": 0.5, "fundamental": 0.3, "sentiment": 0.0, "graph": 0.2}
            total = 1.0

        state.weights = {k: round(v / total, 4) for k, v in w.items()}
        state.log(f"  Final weights: {state.weights}")
        return state

    def _assemble_final_result(self, state: AgentState, tech_signal: Any, fund_signal: Any) -> None:
        sentiment_signal = float(
            state.agent_outputs.get("sentiment", {}).get("sentiment_signal", 0.0)
        )
        graph_out = state.agent_outputs.get("graph", {})

        scores: list[tuple[float, float, float, str]] = []

        if tech_signal.is_valid():
            tech_score = 1.0 if tech_signal.direction == "UP" else 0.0
            tft_direction = tech_signal.direction
            tft_confidence = float(tech_signal.confidence)

            # Sentiment reinforces or weakens technical signal confidence.
            if abs(sentiment_signal) > 0.2:
                if sentiment_signal > 0 and tft_direction == "UP":
                    combined_confidence = min(tft_confidence * 1.15, 1.0)
                elif sentiment_signal < 0 and tft_direction == "DOWN":
                    combined_confidence = min(tft_confidence * 1.15, 1.0)
                else:
                    combined_confidence = tft_confidence * 0.85
            else:
                combined_confidence = tft_confidence

            scores.append(
                (
                    float(state.weights.get("technical", 0.0)),
                    tech_score,
                    combined_confidence,
                    "technical",
                )
            )
            q50 = tech_signal.quantiles.get("0.5")
            if q50 is not None:
                state.final_return_7d = float(q50) * 100.0

        if fund_signal.is_valid():
            fund_score = float(fund_signal.normalized_health_score)
            scores.append(
                (
                    float(state.weights.get("fundamental", 0.0)),
                    fund_score,
                    float(fund_signal.confidence),
                    "fundamental",
                )
            )

        # Graph score is in [-1, +1]. Convert to [0, 1] so all agents share
        # a common score scale before weighted averaging.
        if isinstance(graph_out, dict) and graph_out:
            g_raw = float(graph_out.get("graph_score", 0.0))
            g_conf = float(graph_out.get("confidence", 0.0))
            g_score = max(0.0, min(1.0, (g_raw + 1.0) / 2.0))
            passes_conf = g_conf >= self.graph_min_confidence
            passes_mag = abs(g_raw) >= self.graph_min_abs_score

            # Keep graph output visible for debugging and analysis, but only
            # include it in the final blend when conviction is strong enough.
            graph_out["used_in_ensemble"] = bool(passes_conf and passes_mag)
            graph_out["gate"] = {
                "min_confidence": self.graph_min_confidence,
                "min_abs_score": self.graph_min_abs_score,
                "passes_confidence": passes_conf,
                "passes_magnitude": passes_mag,
            }

            if graph_out["used_in_ensemble"]:
                scores.append(
                    (
                        float(state.weights.get("graph", 0.0)),
                        g_score,
                        g_conf,
                        "graph",
                    )
                )
            else:
                state.log(
                    "  GraphAgent gated out "
                    f"(conf={g_conf:.3f}, |score|={abs(g_raw):.3f})"
                )

        if not scores:
            state.final_direction = "NEUTRAL"
            state.final_return_7d = 0.0
            state.overall_confidence = round(float(state.data_quality_score) * 0.2, 4)
            return

        weight_sum = sum(w for w, _, _, _ in scores)
        if weight_sum <= 0:
            weight_sum = float(len(scores))
            scores = [(1.0, s, c, n) for _, s, c, n in scores]

        combined_score = sum(w * s for w, s, _, _ in scores) / weight_sum
        combined_agent_conf = sum(w * c for w, _, c, _ in scores) / weight_sum

        if combined_score >= 0.55:
            state.final_direction = "UP"
        elif combined_score <= 0.45:
            state.final_direction = "DOWN"
        else:
            state.final_direction = "NEUTRAL"

        # Blend model confidence with data-quality confidence.
        state.overall_confidence = round(
            float(0.65 * combined_agent_conf + 0.35 * state.data_quality_score), 4
        )