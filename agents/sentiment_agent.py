# ============================================================
# agents/sentiment_agent.py
#
# SentimentAgent — queries pre-computed sentiment scores
# stored in news_articles and returns a per-stock signal.
#
# DOES NOT run the model at inference time.
# All scores computed offline by run_sentiment.ipynb.
# ============================================================

import asyncpg
from datetime import date, datetime, time
from agents.base_agent import BaseAgent
from agents.agent_state import AgentState


class SentimentAgent(BaseAgent):
    """
    Computes weighted sentiment signal from pre-scored articles.

    Signal range: -1.0 (fully negative) to +1.0 (fully positive)
    Recency weighting: newest article = 1.0, oldest = 0.5 (linear)
    Confidence: low but non-zero for sparse samples, linear to 1.0 at 10+ articles
    """

    def __init__(self, db_url: str | None = None):
        super().__init__("sentiment")
        self.db_url = db_url

    def _resolve_reference_timestamp(self, state: AgentState) -> datetime:
        """Return a DB-safe datetime anchor for the sentiment lookback window."""

        def _parse(value: object) -> datetime | None:
            if isinstance(value, datetime):
                return value
            if isinstance(value, date):
                return datetime.combine(value, time.max)
            if isinstance(value, str):
                txt = value.strip()
                if not txt:
                    return None
                txt = txt.replace("Z", "+00:00")
                try:
                    parsed = datetime.fromisoformat(txt)
                except ValueError:
                    return None
                # Keep DB parameter naive to match TIMESTAMP WITHOUT TIME ZONE columns.
                if parsed.tzinfo is not None:
                    parsed = parsed.replace(tzinfo=None)
                if "T" not in txt and len(txt) <= 10:
                    parsed = datetime.combine(parsed.date(), time.max)
                return parsed
            return None

        for candidate in (getattr(state, "seance", None), getattr(state, "created_at", None)):
            parsed = _parse(candidate)
            if parsed is not None:
                return parsed

        return datetime.utcnow()

    async def _fetch_window_rows(
        self,
        state: AgentState,
        anchor_ts: datetime,
        window_days: int,
    ):
        return await self.query_db(
            '''
            SELECT sentiment_score,
                   sentiment_label,
                   published_at
            FROM news_articles
                        WHERE (
                                        ticker = $1
                                 OR ($2 <> '' AND isin_code = $2)
                                 OR UPPER(TRIM(ticker)) = UPPER(TRIM($1))
                                 OR REPLACE(UPPER(ticker), ' ', '') = REPLACE(UPPER($1), ' ', '')
                                    )
              AND sentiment_label IS NOT NULL
              AND published_at <= $3
              AND published_at >= ($3 - ($4 * INTERVAL '1 day'))
            ORDER BY published_at DESC
            ''',
            state.ticker,
                        (state.isin_code or ''),
            anchor_ts,
            window_days,
        )

    async def _fetch_latest_scored_timestamp(
        self,
        state: AgentState,
        upper_bound_ts: datetime,
    ) -> datetime | None:
        # Anti-leak: cap MAX(published_at) at the prediction-time anchor so
        # back-tested calls never see articles published after the date
        # they are evaluating.
        rows = await self.query_db(
            '''
            SELECT MAX(published_at) AS latest_at
            FROM news_articles
                        WHERE (
                                        ticker = $1
                                 OR ($2 <> '' AND isin_code = $2)
                                 OR UPPER(TRIM(ticker)) = UPPER(TRIM($1))
                                 OR REPLACE(UPPER(ticker), ' ', '') = REPLACE(UPPER($1), ' ', '')
                                    )
              AND sentiment_label IS NOT NULL
              AND published_at <= $3
            ''',
            state.ticker,
                        (state.isin_code or ''),
            upper_bound_ts,
        )
        if not rows:
            return None
        latest = rows[0].get('latest_at')
        return latest if isinstance(latest, datetime) else None

    async def _fetch_latest_rows_without_window(
        self,
        state: AgentState,
        upper_bound_ts: datetime,
        limit_rows: int = 30,
    ):
        # Anti-leak: same cap as _fetch_latest_scored_timestamp. Without
        # this filter the final fallback would happily return tomorrow's
        # articles when back-testing a 2023 prediction.
        return await self.query_db(
            '''
            SELECT sentiment_score,
                   sentiment_label,
                   published_at
            FROM news_articles
            WHERE (
                    ticker = $1
                 OR ($2 <> '' AND isin_code = $2)
                 OR UPPER(TRIM(ticker)) = UPPER(TRIM($1))
                 OR REPLACE(UPPER(ticker), ' ', '') = REPLACE(UPPER($1), ' ', '')
                  )
              AND sentiment_label IS NOT NULL
              AND published_at <= $3
            ORDER BY published_at DESC NULLS LAST
            LIMIT $4
            ''',
            state.ticker,
            (state.isin_code or ''),
            upper_bound_ts,
            limit_rows,
        )

    async def run(self, state: AgentState) -> AgentState:
        state.log("SentimentAgent started")

        WINDOW_CANDIDATES = [30, 60, 90, 180, 365]
        MIN_ARTICLES_FOR_SIGNAL = 3
        reference_ts = self._resolve_reference_timestamp(state)
        selected_rows = []
        selected_window = WINDOW_CANDIDATES[0]
        used_anchor = reference_ts
        fallback_path: list[str] = []

        try:
            for win in WINDOW_CANDIDATES:
                rows = await self._fetch_window_rows(state, reference_ts, win)
                fallback_path.append(f"ref@{win}d={len(rows)}")
                if len(rows) > len(selected_rows):
                    selected_rows = rows
                    selected_window = win
                    used_anchor = reference_ts
                if len(rows) >= MIN_ARTICLES_FOR_SIGNAL:
                    selected_rows = rows
                    selected_window = win
                    used_anchor = reference_ts
                    break

            # If reference-based windows are empty, anchor on latest scored article.
            # Anti-leak: pass reference_ts as upper bound so the fallback never
            # peeks at articles published after the date being evaluated.
            if len(selected_rows) == 0:
                latest_ts = await self._fetch_latest_scored_timestamp(
                    state,
                    upper_bound_ts=reference_ts,
                )
                if latest_ts is not None:
                    state.log("  No recent rows on reference date; trying latest-article anchor")
                    for win in WINDOW_CANDIDATES:
                        rows = await self._fetch_window_rows(state, latest_ts, win)
                        fallback_path.append(f"latest@{win}d={len(rows)}")
                        if len(rows) > len(selected_rows):
                            selected_rows = rows
                            selected_window = win
                            used_anchor = latest_ts
                        if len(rows) >= MIN_ARTICLES_FOR_SIGNAL:
                            selected_rows = rows
                            selected_window = win
                            used_anchor = latest_ts
                            break

            # Final fallback for sparse/missing timestamps: use latest available scored rows.
            # Same anti-leak cap applies.
            if len(selected_rows) == 0:
                rows = await self._fetch_latest_rows_without_window(
                    state,
                    upper_bound_ts=reference_ts,
                    limit_rows=30,
                )
                fallback_path.append(f"latest_any={len(rows)}")
                if rows:
                    selected_rows = rows
                    selected_window = 999
                    used_anchor = reference_ts

            n = len(selected_rows)
            state.log(f"  Articles selected: {n} (window={selected_window}d)")

            if n == 0:
                state.agent_outputs['sentiment']     = _empty(fallback_path=fallback_path)
                state.confidence_scores['sentiment'] = 0.0
                state.log("  No articles — signal = 0.0")
                return state

            scores = [float(r['sentiment_score']) for r in selected_rows]
            labels = [r['sentiment_label'] for r in selected_rows]

            # Recency weights: index 0 = most recent
            weights = [1.0 - 0.5 * (i / max(n - 1, 1)) for i in range(n)]
            signal  = round(
                sum(s * w for s, w in zip(scores, weights)) / sum(weights),
                4
            )

            pos_ratio  = round(labels.count('positive') / n, 3)
            neg_ratio  = round(labels.count('negative') / n, 3)
            neu_ratio  = round(labels.count('neutral')  / n, 3)
            dominant   = max(['positive','negative','neutral'],
                             key=lambda l: labels.count(l))
            # Keep confidence small but non-zero for 1-2 articles.
            base_conf = min(max(n / 10, 0.0), 1.0)
            if selected_window > 180:
                staleness_mult = 0.75
            elif selected_window > 90:
                staleness_mult = 0.85
            elif selected_window > 60:
                staleness_mult = 0.92
            else:
                staleness_mult = 1.0
            confidence = round(base_conf * staleness_mult, 3) if n > 0 else 0.0

            result = {
                'sentiment_signal':   signal,
                'dominant_label':     dominant,
                'article_count':      n,
                'pos_ratio':          pos_ratio,
                'neg_ratio':          neg_ratio,
                'neu_ratio':          neu_ratio,
                'most_recent_score':  scores[0],
                'window_days':        selected_window,
                'anchor_timestamp':   used_anchor.isoformat(),
                'fallback_used':      (selected_window != 30) or (used_anchor != reference_ts),
                'fallback_path':      fallback_path,
                'confidence':         confidence,
            }

            state.agent_outputs['sentiment']     = result
            state.confidence_scores['sentiment'] = confidence

            state.log(f"  Signal={signal:+.4f}  "
                      f"Dominant={dominant}  Conf={confidence:.3f}")

        except Exception as e:
            state.log(f"  Error: {e}")
            state.agent_outputs['sentiment']     = _empty()
            state.confidence_scores['sentiment'] = 0.0

        state.log("SentimentAgent done")
        return state


def _empty(fallback_path: list[str] | None = None) -> dict:
    return {
        'sentiment_signal': 0.0, 'dominant_label': 'neutral',
        'article_count': 0, 'pos_ratio': 0.0, 'neg_ratio': 0.0,
        'neu_ratio': 0.0, 'most_recent_score': 0.0,
        'window_days': 30,
        'anchor_timestamp': None,
        'fallback_used': False,
        'fallback_path': fallback_path or [],
        'confidence': 0.0,
    }
