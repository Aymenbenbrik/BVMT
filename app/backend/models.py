"""
Pydantic response models for all FastAPI endpoints.
These define the JSON shape the React frontend receives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── /stocks ──────────────────────────────────────────────────────────

class StockListItem(BaseModel):
    ticker: str
    isin_code: Optional[str] = None
    company_type: Optional[str] = None
    sector: Optional[str] = None
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    cached: bool = False
    direction: Optional[str] = None
    confidence: Optional[float] = None
    final_score: Optional[float] = None
    expected_move_pct: Optional[float] = None
    current_close: Optional[float] = None


class StockListResponse(BaseModel):
    stocks: List[StockListItem]
    total: int


# ── /market-overview ─────────────────────────────────────────────────

class MarketOverviewResponse(BaseModel):
    stocks: List[StockListItem]
    total: int
    bull_count: int = 0
    bear_count: int = 0
    hold_count: int = 0
    uncached_count: int = 0
    average_confidence: float = 0.0
    market_mood: str = "UNKNOWN"


# ── /prediction/{ticker} ────────────────────────────────────────────

class PredictionMetadata(BaseModel):
    ticker: str
    isin_code: Optional[str] = None
    company_type: Optional[str] = None
    run_timestamp: Optional[str] = None
    cache_hit: bool = False
    computation_time_seconds: float = 0.0


class DataQualitySection(BaseModel):
    rows_available: int = 0
    latest_date: Optional[str] = None
    null_count: int = 0
    news_article_count: int = 0
    financial_statements_available: bool = False
    data_score: float = 0.0
    years_of_data: int = 0
    skip_flags: List[str] = Field(default_factory=list)


class PriceHorizonPoint(BaseModel):
    day_number: int
    predicted_price: float
    predicted_return_pct: Optional[float] = None


class TopFeature(BaseModel):
    feature_name: str
    current_value: float = 0.0
    z_score: float = 0.0
    interpretation: str = ""


class TechnicalSection(BaseModel):
    direction: str = "UNKNOWN"
    confidence: float = 0.0
    signal_str: str = ""
    price_horizon: List[PriceHorizonPoint] = Field(default_factory=list)
    top_features: List[TopFeature] = Field(default_factory=list)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    expected_move_pct: Optional[float] = None
    current_close: Optional[float] = None


class RatioDetail(BaseModel):
    value: float
    year: Optional[str] = None
    benchmark: Optional[str] = None


class FundamentalSection(BaseModel):
    health_class: str = "UNKNOWN"
    confidence: float = 0.0
    health_score: float = 0.0
    normalized_health_score: float = 0.0
    ratios: Dict[str, Any] = Field(default_factory=dict)
    period: Optional[str] = None
    company_type: Optional[str] = None
    model_info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class RecentArticle(BaseModel):
    title: Optional[str] = None
    date: Optional[str] = None
    sentiment_score: float = 0.0
    source: str = "ilboursa.com"


class SentimentSection(BaseModel):
    score: float = 0.0
    label: str = "NEUTRAL"
    confidence: float = 0.0
    article_count: int = 0
    date_range: Dict[str, Any] = Field(default_factory=dict)
    recent_articles: List[RecentArticle] = Field(default_factory=list)
    model_info: Dict[str, Any] = Field(default_factory=dict)


class NeighborInfo(BaseModel):
    ticker: str
    correlation: float = 0.0
    direction: Optional[str] = None
    influence_type: str = "concurrent"
    explanation: str = ""
    relation_score: float = 0.0
    sector: Optional[str] = None


class GraphSection(BaseModel):
    signal: str = "NEUTRAL"
    confidence: float = 0.0
    neighborhood_score: float = 0.0
    top_neighbors: List[NeighborInfo] = Field(default_factory=list)
    graph_stats: Dict[str, Any] = Field(default_factory=dict)
    confidence_gating_applied: bool = False
    class_probs: Dict[str, float] = Field(default_factory=dict)


class XAISection(BaseModel):
    technical_explanation: Dict[str, Any] = Field(default_factory=dict)
    fundamental_explanation: Dict[str, Any] = Field(default_factory=dict)
    graph_explanation: Dict[str, Any] = Field(default_factory=dict)


class OrchestrationSection(BaseModel):
    final_direction: str = "NEUTRAL"
    final_confidence: float = 0.0
    final_score: float = 0.0
    signal_str: str = ""
    agent_weights: Dict[str, float] = Field(default_factory=dict)
    agent_contributions: Dict[str, Any] = Field(default_factory=dict)
    agents_failed: List[str] = Field(default_factory=list)
    decision_explanation: str = ""


class AIArticleSection(BaseModel):
    content: Optional[str] = None
    generated_at: Optional[str] = None
    word_count: int = 0


class PredictionResponse(BaseModel):
    metadata: PredictionMetadata
    data_quality: DataQualitySection
    technical: TechnicalSection
    fundamental: FundamentalSection
    sentiment: SentimentSection
    graph: GraphSection
    xai: XAISection
    orchestration: OrchestrationSection
    ai_article: AIArticleSection
    explanation_facts: Dict[str, Any] = Field(default_factory=dict)
    execution_log: List[str] = Field(default_factory=list)


# ── /graph-data ──────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    label: str
    company_type: Optional[str] = None
    sector: Optional[str] = None
    final_direction: Optional[str] = None
    final_confidence: Optional[float] = None
    final_score: Optional[float] = None


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float = 0.0
    correlation_type: str = "positive"
    strength_label: str = "WEAK"


class GraphMetadataInfo(BaseModel):
    total_nodes: int = 0
    total_edges: int = 0
    computation_date: Optional[str] = None
    correlation_window_days: int = 120


class GraphDataResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    graph_metadata: GraphMetadataInfo


# ── /stock-history/{ticker} ──────────────────────────────────────────

class PricePoint(BaseModel):
    date: str
    close: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    daily_return: Optional[float] = None
    is_predicted: bool = False


class StockHistoryResponse(BaseModel):
    ticker: str
    history: List[PricePoint]
    predicted: List[PricePoint] = Field(default_factory=list)


# ── /cache/status ────────────────────────────────────────────────────

class CacheStatusResponse(BaseModel):
    cached_count: int = 0
    ttl_seconds: int = 7200
    entries: Dict[str, Any] = Field(default_factory=dict)


# ── Coverage report ─────────────────────────────────────────────────

class CoverageReportResponse(BaseModel):
    total_before: int = 0
    remaining_count: int = 0
    removed_count: int = 0
    removed_tickers: List[str] = Field(default_factory=list)
