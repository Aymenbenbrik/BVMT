# main.py
# ──────────────────────────────────────────────────────────────────────
# PURPOSE:  **Entry point** of the BVMT Multi-Agent Platform.
#
# WHAT IS AN ENTRY POINT?
#   It's the file you run to start the whole application:
#       python main.py
#   This file wires everything together: it creates the shared state,
#   picks the right orchestrator, and kicks off the pipeline.
#
# WHY KEEP IT THIN?
#   main.py should do as LITTLE logic as possible.  All the real work
#   happens inside the agents.  This makes the project:
#     • Testable — you can test agents without running main.py
#     • Flexible — you can swap the CLI for a web UI later
#
# FINAL VERSION (Week 11):
#   This will become a FastAPI web server or Streamlit dashboard.
#   For now it's a simple CLI demo.
#
# USAGE:
#   python main.py
# ──────────────────────────────────────────────────────────────────────

import asyncio
import logging

from agents.orchestrator import OrchestratorAgent


async def main():
    """Run a simple async demo of the Day 5 multi-agent pipeline."""
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    print("=" * 60)
    print("  BVMT Multi-Agent Platform — Day 5 Demo")
    print("=" * 60)
    print()

    ticker = "AMEN BANK"
    print(f"Ticker selected: {ticker}")
    print()

    orchestrator = OrchestratorAgent()
    result = await orchestrator.run(ticker)

    print()
    print("-" * 60)
    print(f"Ticker:             {result.ticker}")
    print(f"Request ID:         {result.request_id}")
    print(f"Price rows:         {result.price_rows_available}")
    print(f"News articles:      {result.news_articles_count}")
    print(f"Data quality score: {result.data_quality_score:.3f}")
    print(f"Weights:            {result.weights}")
    print(f"Final direction:    {result.final_direction}")
    print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
