/**
 * TypeScript interfaces matching all backend Pydantic models.
 */

export interface StockListItem {
  ticker: string;
  isin_code: string | null;
  company_type: string | null;
  sector: string | null;
  first_seen: string | null;
  last_seen: string | null;
  cached: boolean;
  direction: string | null;
  confidence: number | null;
  final_score: number | null;
  expected_move_pct?: number | null;
  current_close?: number | null;
}

export interface StockListResponse {
  stocks: StockListItem[];
  total: number;
}

export interface MarketOverviewResponse {
  stocks: StockListItem[];
  total: number;
  bull_count: number;
  bear_count: number;
  hold_count: number;
  uncached_count: number;
  average_confidence: number;
  market_mood: string;
}

// ── Prediction ──────────────────────────────────────────────────

export interface PredictionMetadata {
  ticker: string;
  isin_code: string | null;
  company_type: string | null;
  run_timestamp: string | null;
  cache_hit: boolean;
  computation_time_seconds: number;
}

export interface DataQualitySection {
  rows_available: number;
  latest_date: string | null;
  null_count: number;
  news_article_count: number;
  financial_statements_available: boolean;
  data_score: number;
  years_of_data: number;
  skip_flags: string[];
}

export interface PriceHorizonPoint {
  day_number: number;
  predicted_price: number;
  predicted_return_pct: number | null;
}

export interface TopFeature {
  feature_name: string;
  current_value: number;
  z_score: number;
  interpretation: string;
}

export interface TechnicalSection {
  direction: string;
  confidence: number;
  signal_str: string;
  price_horizon: PriceHorizonPoint[];
  top_features: TopFeature[];
  model_info: Record<string, any>;
  error: string | null;
  expected_move_pct?: number | null;
  current_close?: number | null;
}

export interface FundamentalSection {
  health_class: string;
  confidence: number;
  health_score: number;
  normalized_health_score: number;
  ratios: Record<string, any>;
  period: string | null;
  company_type: string | null;
  model_info: Record<string, any>;
  error: string | null;
}

export interface RecentArticle {
  title: string | null;
  date: string | null;
  sentiment_score: number;
  source: string;
}

export interface SentimentSection {
  score: number;
  label: string;
  confidence: number;
  article_count: number;
  date_range: Record<string, any>;
  recent_articles: RecentArticle[];
  model_info: Record<string, any>;
}

export interface NeighborInfo {
  ticker: string;
  correlation: number;
  direction: string | null;
  influence_type: string;
  explanation: string;
  relation_score: number;
  sector: string | null;
}

export interface GraphSection {
  signal: string;
  confidence: number;
  neighborhood_score: number;
  top_neighbors: NeighborInfo[];
  graph_stats: Record<string, any>;
  confidence_gating_applied: boolean;
  class_probs: Record<string, number>;
}

export interface XAISection {
  technical_explanation: Record<string, any>;
  fundamental_explanation: Record<string, any>;
  graph_explanation: Record<string, any>;
}

export interface OrchestrationSection {
  final_direction: string;
  final_confidence: number;
  final_score: number;
  signal_str: string;
  agent_weights: Record<string, number>;
  agent_contributions: Record<string, any>;
  agents_failed: string[];
  decision_explanation: string;
}

export interface AIArticleSection {
  content: string | null;
  generated_at: string | null;
  word_count: number;
}

export interface PredictionResponse {
  metadata: PredictionMetadata;
  data_quality: DataQualitySection;
  technical: TechnicalSection;
  fundamental: FundamentalSection;
  sentiment: SentimentSection;
  graph: GraphSection;
  xai: XAISection;
  orchestration: OrchestrationSection;
  ai_article: AIArticleSection;
  explanation_facts: Record<string, any>;
  execution_log: string[];
}

// ── Graph Data ──────────────────────────────────────────────────

export interface GraphNode {
  id: string;
  label: string;
  company_type: string | null;
  sector: string | null;
  final_direction: string | null;
  final_confidence: number | null;
  final_score: number | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  correlation_type: string;
  strength_label: string;
}

export interface GraphMetadataInfo {
  total_nodes: number;
  total_edges: number;
  computation_date: string | null;
  correlation_window_days: number;
}

export interface GraphDataResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  graph_metadata: GraphMetadataInfo;
}

// ── Stock History ───────────────────────────────────────────────

export interface PricePoint {
  date: string;
  close: number;
  open: number | null;
  high: number | null;
  low: number | null;
  volume: number | null;
  daily_return: number | null;
  is_predicted: boolean;
}

export interface StockHistoryResponse {
  ticker: string;
  history: PricePoint[];
  predicted: PricePoint[];
}

// ── Cache Status ────────────────────────────────────────────────

export interface CacheStatusResponse {
  cached_count: number;
  ttl_seconds: number;
  entries: Record<string, any>;
}
