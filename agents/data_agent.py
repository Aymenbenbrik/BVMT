"""
DataAgent: Loads price, news, and financial data for a given stock.
Computes a data_quality_score that the OrchestratorAgent uses
to adjust the weights of the other agents.
"""

from datetime import date, timedelta

from agents.agent_state import AgentState
from agents.base_agent import BaseAgent


class DataAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataAgent")

    async def run(self, state: AgentState) -> AgentState:
        state.log(f"DataAgent: loading data for {state.ticker}")
        isin_key = (state.isin_code or "").strip()

        try:
            # 1. Count price rows available for this stock
            rows = await self.query_db(
                """
                SELECT COUNT(*) as cnt
                FROM daily_prices
                WHERE ticker = $1 OR ($2 <> '' AND isin_code = $2)
                """,
                state.ticker,
                isin_key,
            )
            state.price_rows_available = rows[0]["cnt"] if rows else 0
            state.log(f"  Price rows: {state.price_rows_available}")

            # 2. Count years of data
            yr_rows = await self.query_db(
                """
                SELECT COUNT(DISTINCT EXTRACT(YEAR FROM seance)) as yrs
                FROM daily_prices
                WHERE ticker = $1 OR ($2 <> '' AND isin_code = $2)
                """,
                state.ticker,
                isin_key,
            )
            state.years_of_price_data = int(yr_rows[0]["yrs"]) if yr_rows else 0

            # 3. Count news articles
            news_rows = await self.query_db(
                """
                SELECT COUNT(*) as cnt
                FROM news_articles
                WHERE ticker = $1 OR ($2 <> '' AND isin_code = $2)
                """,
                state.ticker,
                isin_key,
            )
            state.news_articles_count = news_rows[0]["cnt"] if news_rows else 0
            state.log(f"  News articles: {state.news_articles_count}")

            # 4. Check for recent financial statements
            # 'Recent' = period_end_date within the last 18 months
            recent_cutoff = date.today() - timedelta(days=548)
            fin_rows = await self.query_db(
                """
                SELECT COUNT(*) as cnt
                FROM financial_statements
                WHERE (ticker = $1 OR ($2 <> '' AND isin_code = $2))
                AND period_end_date >= $3
                """,
                state.ticker,
                isin_key,
                recent_cutoff,
            )
            state.has_recent_financials = (fin_rows[0]["cnt"] > 0) if fin_rows else False
            state.log(f"  has_recent_financials: {state.has_recent_financials}")
            state.log(f"  financial recency cutoff: {recent_cutoff}")

            # 5. Compute data quality score
            state.data_quality_score = self._compute_quality(state)
            state.log(f"  Data quality score: {state.data_quality_score:.3f}")

            state.agent_outputs["data"] = {
                "price_rows": state.price_rows_available,
                "years": state.years_of_price_data,
                "news_count": state.news_articles_count,
                "has_financials": state.has_recent_financials,
                "quality_score": state.data_quality_score,
                "status": "success",
            }

        except Exception as e:
            state.log(f"  DataAgent ERROR: {e}")
            state.agent_outputs["data"] = {"status": "error", "error": str(e)}

        return state

    def _compute_quality(self, state: AgentState) -> float:
        """
        50% -- price data  (full marks at 2000+ rows)
        30% -- news        (full marks at 20+ articles)
        20% -- financials  (binary: True=0.20, False=0.00)
        """
        price_score = min(state.price_rows_available / 2000, 1.0)
        news_score = min(state.news_articles_count / 20, 1.0)
        fin_score = 1.0 if state.has_recent_financials else 0.0
        total = round(0.50 * price_score + 0.30 * news_score + 0.20 * fin_score, 4)
        state.log(
            f"  Quality: price={price_score:.2f} news={news_score:.2f} "
            f"fin={fin_score:.2f} -> {total}"
        )
        return total
