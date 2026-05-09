import pytest
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))


import psycopg2
from agents.orchestrator import OrchestratorAgent
from agents.agent_state import AgentState


def get_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'), port=int(os.getenv('DB_PORT', 5432)),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'))

# T1 — price data
def test_prices_loaded():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute('SELECT COUNT(*) FROM daily_prices')
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT ticker) FROM daily_prices")
        stocks = cur.fetchone()[0]
    conn.close()
    assert total >= 100_000, f'Expected 100k+ rows, got {total}'
    assert stocks >= 60,     f'Expected 60+ stocks, got {stocks}'

# T2 — indicators
def test_indicators_computed():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM computed_indicators
            WHERE rsi_14 IS NOT NULL AND ma_20 IS NOT NULL""")
        ok = cur.fetchone()[0]
    conn.close()
    assert ok >= 50_000, f'Expected 50k+ indicator rows, got {ok}'

# T3 — news
def test_news_loaded():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute('SELECT COUNT(*) FROM news_articles')
        total = cur.fetchone()[0]
        cur.execute("""
            SELECT COUNT(*) FROM news_articles
            WHERE published_at >= '2024-01-01'""")
        recent = cur.fetchone()[0]
    conn.close()
    assert total >= 5_000,  f'Expected 5k+ news rows, got {total}'
    assert recent >= 200,   f'Expected 200+ recent articles, got {recent}'

# T4 — financials
def test_financials_available():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(DISTINCT ticker) FROM financial_statements
            WHERE company_type = 'bank'
            AND total_assets IS NOT NULL""")
        banks = cur.fetchone()[0]
    conn.close()
    assert banks >= 5, f'Expected 5+ banks with financial data, got {banks}'

# T5 — full pipeline
def test_pipeline_runs():
    orch = OrchestratorAgent()
    result = asyncio.run(orch.run('AMEN BANK'))
    assert result is not None
    assert hasattr(result, 'data_quality_score') or 'data_quality_score' in result
