/**
 * Axios HTTP client + typed API functions for all backend endpoints.
 */
import axios from 'axios';
import type {
  StockListResponse,
  MarketOverviewResponse,
  PredictionResponse,
  GraphDataResponse,
  StockHistoryResponse,
  CacheStatusResponse,
} from '../types';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 120_000, // 2 min — agent pipeline can be slow
});

// ── Stocks ──────────────────────────────────────────────────────

export async function fetchStocks(): Promise<StockListResponse> {
  const { data } = await api.get<StockListResponse>('/stocks');
  return data;
}

export async function fetchMarketOverview(): Promise<MarketOverviewResponse> {
  const { data } = await api.get<MarketOverviewResponse>('/market-overview');
  return data;
}

export async function fetchStockHistory(ticker: string): Promise<StockHistoryResponse> {
  const { data } = await api.get<StockHistoryResponse>(`/stock-history/${encodeURIComponent(ticker)}`);
  return data;
}

// ── Prediction ──────────────────────────────────────────────────

export async function fetchPrediction(ticker: string): Promise<PredictionResponse> {
  const { data } = await api.get<PredictionResponse>(`/prediction/${encodeURIComponent(ticker)}`);
  return data;
}

export async function refreshTicker(ticker: string): Promise<PredictionResponse> {
  const { data } = await api.post<PredictionResponse>(`/cache/refresh/${encodeURIComponent(ticker)}`);
  return data;
}

// ── Graph ───────────────────────────────────────────────────────

export async function fetchGraphData(threshold?: number): Promise<GraphDataResponse> {
  const params = threshold !== undefined ? { threshold } : {};
  const { data } = await api.get<GraphDataResponse>('/graph-data', { params });
  return data;
}

// ── Cache ───────────────────────────────────────────────────────

export async function fetchCacheStatus(): Promise<CacheStatusResponse> {
  const { data } = await api.get<CacheStatusResponse>('/cache/status');
  return data;
}

export async function triggerPrecompute(): Promise<void> {
  await api.post('/cache/precompute');
}

// ── Explain (on-demand XAI narratives) ──────────────────────────────
export interface XaiNarrativesRequest {
  ticker: string;
  explain_version?: string;
  technical: Record<string, any>;
  fundamental: Record<string, any>;
  graph: Record<string, any>;
}

export interface XaiNarrativesResponse {
  technical: string;
  fundamental: string;
  graph: string;
  model_used: string;
  generated_at: string;
  cached: boolean;
}

export async function generateXaiNarratives(
  payload: XaiNarrativesRequest,
): Promise<XaiNarrativesResponse> {
  const { data } = await api.post<XaiNarrativesResponse>('/explain/xai', payload);
  return data;
}

export default api;
