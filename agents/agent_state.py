"""
AgentState: Shared state object passed between all agents.

Think of it as a project file that every agent reads and writes.
The OrchestratorAgent creates it at the start of each analysis,
passes it to each agent in order, and reads the final result from it.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from datetime import datetime


@dataclass
class AgentState:
    # -- Request identity -------------------------------------------------
    ticker: str = ""            # e.g. "AMEN BANK"
    isin_code: str = ""         # e.g. "TN0003400058"
    request_id: str = ""        # unique ID for this analysis run
    user_role: str = "investor" # "investor", "analyst", "admin"
    created_at: str = ""

    # -- Data quality flags (filled by DataAgent) -------------------------
    price_rows_available: int = 0
    news_articles_count: int = 0
    data_quality_score: float = 0.0  # 0.0 to 1.0
    has_recent_financials: bool = False
    years_of_price_data: int = 0

    # -- Agent weights (adjusted by OrchestratorAgent based on data quality)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "technical": 0.34,
        "fundamental": 0.26,
        "sentiment": 0.20,
        "graph": 0.20,
    })

    # -- Skip flags (set when data is insufficient for an agent) ----------
    skip_sentiment: bool = False
    skip_fundamental: bool = False
    use_cnn_lstm_fallback: bool = False

    # -- Agent outputs (each agent writes its result here) -----------------
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    # Keys will be: "data", "technical", "fundamental", "sentiment",
    #               "graph", "risk", "xai"

    # -- Confidence scores per agent (0.0 to 1.0) -------------------------
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    # -- Final assembled result (filled at end of pipeline) ---------------
    final_direction: str = ""   # "UP", "DOWN", "NEUTRAL"
    final_return_7d: float = 0.0 # predicted 7-day return %
    overall_confidence: float = 0.0
    risk_score: float = 0.0      # 0.0 = low risk, 1.0 = high risk

    # -- Execution log (every step is recorded here) ----------------------
    execution_log: List[str] = field(default_factory=list)

    # -- Graph/correlation data --------------------------------------------
    sector_shock_detected: bool = False
    attention_weights: Dict[str, float] = field(default_factory=dict)

    def log(self, message: str):
        """Add a timestamped entry to the execution log."""
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        self.execution_log.append(entry)
        print(entry)  # also print to console during development

    def to_dict(self) -> dict:
        """Serialize state to a plain dict for JSON storage."""
        import dataclasses
        return dataclasses.asdict(self)
