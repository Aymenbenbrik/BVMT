# tests/test_data_agent.py
# ──────────────────────────────────────────────────────────────────────
# PURPOSE:  Unit tests for the DataAgent.
#
# WHY WRITE TESTS?
#   Tests let you verify each agent works correctly in isolation,
#   without running the whole system.  When you change code later,
#   running tests instantly tells you if you broke something.
#
# HOW TO RUN:
#   From the project root:
#       python -m pytest tests/ -v
#
# WHAT EACH TEST DOES:
#   test_data_agent_missing_file
#       → Checks that a helpful error is stored when the CSV doesn't exist.
#
#   test_data_agent_state_default
#       → Checks that a fresh AgentState has the expected default values.
# ──────────────────────────────────────────────────────────────────────

import asyncio
import os
import sys
import types

# Add the project root to Python's module search path so we can
# import from "agents" even when running from the tests/ folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.agent_state import AgentState
from agents.data_agent import DataAgent


def test_data_agent_run_populates_quality_fields():
    """DataAgent should fill quality-related fields from query results."""
    state = AgentState(ticker="AMEN BANK")
    agent = DataAgent()

    async def fake_query_db(self, sql: str, *params):
        if "COUNT(DISTINCT EXTRACT(YEAR FROM seance))" in sql:
            return [{"yrs": 8}]
        if "FROM daily_prices" in sql:
            return [{"cnt": 1800}]
        if "FROM news_articles" in sql:
            return [{"cnt": 12}]
        return []

    agent.query_db = types.MethodType(fake_query_db, agent)
    result = asyncio.run(agent.run(state))

    assert result.price_rows_available == 1800
    assert result.years_of_price_data == 8
    assert result.news_articles_count == 12
    assert result.has_recent_financials is False
    assert 0.0 <= result.data_quality_score <= 1.0
    assert result.agent_outputs.get("data", {}).get("status") == "success"


def test_agent_state_day5_defaults():
    """A fresh Day 5 AgentState should have expected defaults."""
    state = AgentState()
    assert state.ticker == ""
    assert state.user_role == "investor"
    assert state.price_rows_available == 0
    assert state.news_articles_count == 0
    assert state.data_quality_score == 0.0
    assert state.weights == {
        "technical": 0.42,
        "fundamental": 0.31,
        "sentiment": 0.27,
    }
    assert state.agent_outputs == {}
    assert state.execution_log == []


def test_data_agent_compute_quality_formula():
    """Quality formula should match Day 5 weighting rules."""
    agent = DataAgent()
    state = AgentState(
        price_rows_available=1000,   # price_score = 0.5
        news_articles_count=10,      # news_score = 0.5
        has_recent_financials=False, # fin_score = 0.0
    )
    # 0.50*0.5 + 0.30*0.5 + 0.20*0 = 0.40
    assert agent._compute_quality(state) == 0.4


if __name__ == "__main__":
    # Allow running directly: python tests/test_data_agent.py
    test_data_agent_run_populates_quality_fields()
    print("✔ test_data_agent_run_populates_quality_fields passed")

    test_agent_state_day5_defaults()
    print("✔ test_agent_state_day5_defaults passed")

    test_data_agent_compute_quality_formula()
    print("✔ test_data_agent_compute_quality_formula passed")

    print("\nAll tests passed!")
