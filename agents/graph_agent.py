# ============================================================
# agents/graph_agent.py
#
# GraphAgent — runs the trained GATBVMT model at inference
# time to produce a graph-adjusted direction score for the
# target stock, incorporating neighbourhood influence from
# correlated stocks.
#
# DOES NOT retrain the model. Loads the checkpoint once and
# caches it. At inference time, builds node features for all
# 68 stocks, runs one GAT forward pass, returns the score
# for the target stock only.
#
# WHY FULL GRAPH AT INFERENCE:
#   GAT requires the full graph — you cannot run it on a
#   single node in isolation. All 68 nodes must be present
#   for the attention mechanism to aggregate neighbourhood
#   information correctly.
#
# INPUT CONTRACT (same order as training):
#   x[:, 0] = rsi_norm
#   x[:, 1] = volatility_norm
#   x[:, 2] = ma_ratio_norm
#   x[:, 3] = daily_return_norm
#   x[:, 4] = sentiment_score
#   x[:, 5] = fundamental_health_norm
#   x[:, 6] = data_quality_score
# ============================================================

import torch
import numpy as np
import pandas as pd
import asyncpg
import json
import math
import re
from pathlib import Path
from typing import Any
from agents.base_agent import BaseAgent
from agents.agent_state import AgentState

# Import the GAT model class
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.gat_bvmt import GATBVMT, load_gat

GRAPH_DIR  = Path(__file__).parent.parent / 'data' / 'graph'
PRICES_DIR = Path(__file__).parent.parent / 'data' / 'prices'
MODELS_DIR = Path(__file__).parent.parent / 'models'
GAT_CKPT   = MODELS_DIR / 'gat_bvmt_best.ckpt'
FEATURES_DIR = Path(__file__).parent.parent / 'data' / 'features'
TFT_FEATURES_PATH = FEATURES_DIR / 'tft_features.csv'
TFT_MAPPING_PATH = FEATURES_DIR / 'ticker_mapping.csv'

# Class index → label mapping (matches training)
IDX_TO_LABEL = {0: 'DOWN', 1: 'FLAT', 2: 'UP'}

# Class-prior logit adjustment to reduce FLAT dominance.
# adjusted_logit = raw_logit - tau * log(class_prior)
LOGIT_ADJUST_TAU = 0.90
DEFAULT_CLASS_PRIORS = {
    'DOWN': 0.31,
    'FLAT': 0.38,
    'UP': 0.31,
}

# Cache the loaded model and graph across calls
# (avoid reloading 4MB checkpoint for every stock)
_MODEL_CACHE: GATBVMT       = None
_EDGE_INDEX:  torch.Tensor  = None
_EDGE_WEIGHT: torch.Tensor  = None
_TICKER_IDX:  dict          = None
_CLASS_PRIORS: torch.Tensor = None
_ADJ_MATRIX:  np.ndarray    = None
_TFT_LATEST_FEATURES: dict  = None
_TFT_DQ_SCORES: dict        = None
_TICKER_SECTOR_MAP: dict    = None
_SECTOR_TICKERS: dict       = None


def _stable_seed(text: str) -> int:
    seed = 0
    for index, char in enumerate(text):
        seed = (seed * 131 + (index + 1) * ord(char)) % 4294967291
    return seed or 1


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _sigmoid_norm(value: float, scale: float = 1.0) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(value) * scale))
    except (OverflowError, ValueError, TypeError):
        return 0.5


def _fund_score_from_fields(company_type, roe, roa, cost_income) -> float:
    ctype = str(company_type or 'non_bank').lower().strip()

    def _to_float(val):
        try:
            if val is None:
                return None
            return float(val)
        except (TypeError, ValueError):
            return None

    roe_val = _to_float(roe)
    roa_val = _to_float(roa)
    cir_val = _to_float(cost_income)

    score = 0.0
    if ctype == 'bank':
        if roe_val is not None and roe_val > 10.0:
            score += 1.0
        if roa_val is not None and roa_val > 1.0:
            score += 1.0
        if cir_val is not None and cir_val < 50.0:
            score += 1.0
    else:
        if roe_val is not None and roe_val > 8.0:
            score += 1.0
        if roa_val is not None and roa_val > 2.0:
            score += 1.0
        if cir_val is not None and cir_val < 65.0:
            score += 1.0

    return float(score / 3.0)


def _load_tft_feature_cache():
    """Load latest leakage-safe TFT features and per-ticker data quality."""
    global _TFT_LATEST_FEATURES, _TFT_DQ_SCORES

    if _TFT_LATEST_FEATURES is not None and _TFT_DQ_SCORES is not None:
        return

    _TFT_LATEST_FEATURES = {}
    _TFT_DQ_SCORES = {}

    if _TICKER_IDX is None:
        return

    graph_tickers = set(_TICKER_IDX.keys())
    if not TFT_FEATURES_PATH.exists():
        for ticker in graph_tickers:
            _TFT_DQ_SCORES[ticker] = 0.6
        return

    required_cols = [
        'date', 'ticker', 'ticker_id',
        'close_price', 'rsi_14', 'volatility_20', 'ma_20', 'daily_return',
    ]

    try:
        df_raw = pd.read_csv(TFT_FEATURES_PATH)
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        if missing_cols:
            for ticker in graph_tickers:
                _TFT_DQ_SCORES[ticker] = 0.6
            return

        df = df_raw[required_cols].copy()

        if TFT_MAPPING_PATH.exists() and 'ticker_id' in df.columns:
            mapping = pd.read_csv(TFT_MAPPING_PATH)
            if 'ticker_id' in mapping.columns and 'ticker' in mapping.columns:
                mapping['ticker'] = mapping['ticker'].astype(str).str.strip()
                id_to_ticker = dict(zip(mapping['ticker_id'], mapping['ticker']))
                missing_ticker = (
                    df['ticker'].isna()
                    | (df['ticker'].astype(str).str.strip() == '')
                    | (df['ticker'].astype(str).str.lower() == 'nan')
                )
                if missing_ticker.any():
                    df.loc[missing_ticker, 'ticker'] = df.loc[missing_ticker, 'ticker_id'].map(id_to_ticker)

        df['ticker'] = df['ticker'].astype(str).str.strip()
        df['seance'] = pd.to_datetime(df['date'], errors='coerce')

        numeric_cols = ['close_price', 'rsi_14', 'volatility_20', 'ma_20', 'daily_return']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['seance', 'ticker'])
        df = df[df['ticker'].isin(graph_tickers)].copy()

        df['price_to_ma20'] = np.where(
            (df['ma_20'] > 0) & (df['close_price'] > 0),
            (df['close_price'] / df['ma_20']) - 1.0,
            np.nan,
        )

        df = (df
              .sort_values(['ticker', 'seance'])
              .drop_duplicates(['ticker', 'seance'], keep='last'))

        latest = df.groupby('ticker', as_index=False).tail(1)
        for _, row in latest.iterrows():
            _TFT_LATEST_FEATURES[row['ticker']] = {
                'rsi_14': row.get('rsi_14', np.nan),
                'volatility_20': row.get('volatility_20', np.nan),
                'price_to_ma20': row.get('price_to_ma20', np.nan),
                'daily_return': row.get('daily_return', np.nan),
            }

        coverage_cols = ['rsi_14', 'volatility_20', 'price_to_ma20', 'daily_return', 'close_price']
        for ticker in graph_tickers:
            tdf = df[df['ticker'] == ticker]
            if tdf.empty:
                _TFT_DQ_SCORES[ticker] = 0.6
                continue
            cov_parts = [float(tdf[col].notna().mean()) for col in coverage_cols]
            _TFT_DQ_SCORES[ticker] = _clamp(float(np.mean(cov_parts)), 0.0, 1.0)

    except Exception:
        _TFT_LATEST_FEATURES = {}
        _TFT_DQ_SCORES = {ticker: 0.6 for ticker in graph_tickers}


def _normalize_group_name(value: Any) -> str:
    text = str(value or '').strip()
    if not text or text.lower() in {'nan', 'none', 'null'}:
        return 'UNKNOWN'
    return ' '.join(text.replace('_', ' ').split()).upper()


def _load_sector_map():
    """Load ticker -> sector/group mapping from historical BVMT price files."""
    global _TICKER_SECTOR_MAP, _SECTOR_TICKERS

    if _TICKER_SECTOR_MAP is not None and _SECTOR_TICKERS is not None:
        return

    _TICKER_SECTOR_MAP = {}
    _SECTOR_TICKERS = {}

    if not PRICES_DIR.exists():
        return

    code_to_ticker: dict[str, str] = {}
    scraper_sector_map: dict[str, str] = {}
    scraper_path = Path(__file__).parent.parent / 'scrape_ilboursa.py'
    if scraper_path.exists():
        try:
            text = scraper_path.read_text(encoding='utf-8', errors='ignore')
            current_sector = 'UNKNOWN'
            for line in text.splitlines():
                header_match = re.match(r"\s*#\s*─+\s*(.*?)\s*─+\s*$", line)
                if header_match:
                    current_sector = _normalize_group_name(header_match.group(1))
                    continue
                entry_match = re.match(r"\s*'([^']+)'\s*:\s*'([^']+)'", line)
                if entry_match:
                    ticker_name = entry_match.group(1).strip()
                    code = entry_match.group(2).strip()
                    if ticker_name and code:
                        code_to_ticker[code.upper()] = ticker_name
                        scraper_sector_map[ticker_name] = current_sector
        except Exception:
            code_to_ticker = {}
            scraper_sector_map = {}

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(PRICES_DIR.glob('histo_cotation_*.csv')):
        try:
            df = pd.read_csv(csv_path, sep=';', engine='python')
        except Exception:
            try:
                df = pd.read_csv(csv_path, sep=';', engine='python', encoding='latin1')
            except Exception:
                continue

        columns = {str(col).strip(): str(col).strip() for col in df.columns}
        df = df.rename(columns=columns)
        if 'GROUPE' not in df.columns:
            continue

        if 'VALEUR' in df.columns:
            df['ticker_name'] = df['VALEUR'].astype(str).str.strip()
        else:
            df['ticker_name'] = ''
        if 'CODE' in df.columns:
            code_series = df['CODE'].astype(str).str.strip().str.upper()
            mapped_from_code = code_series.map(code_to_ticker)
            known_graph_tickers = set(_TICKER_IDX.keys()) if _TICKER_IDX else set()
            use_code_map = (~df['ticker_name'].isin(known_graph_tickers)) & mapped_from_code.notna()
            df.loc[use_code_map, 'ticker_name'] = mapped_from_code[use_code_map]
            df['ticker_name'] = df['ticker_name'].where(df['ticker_name'] != '', mapped_from_code)
            df['ticker_name'] = df['ticker_name'].where(df['ticker_name'].notna(), mapped_from_code)
        df['ticker_name'] = df['ticker_name'].astype(str).str.strip()
        df = df[df['ticker_name'] != '']
        if df.empty:
            continue
        frames.append(df[['ticker_name', 'GROUPE']].rename(columns={'ticker_name': 'VALEUR'}).copy())

    for ticker, sector in scraper_sector_map.items():
        sector_norm = _normalize_group_name(sector)
        if ticker:
            _TICKER_SECTOR_MAP[ticker] = sector_norm
            _SECTOR_TICKERS.setdefault(sector_norm, []).append(ticker)

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    combined['VALEUR'] = combined['VALEUR'].astype(str).str.strip()
    combined['GROUPE'] = combined['GROUPE'].apply(_normalize_group_name)
    combined = combined[combined['VALEUR'] != '']

    for ticker, ticker_rows in combined.groupby('VALEUR'):
        if ticker_rows.empty:
            continue
        sector = ticker_rows['GROUPE'].value_counts().idxmax()
        if ticker not in _TICKER_SECTOR_MAP or _TICKER_SECTOR_MAP.get(ticker) == 'UNKNOWN':
            _TICKER_SECTOR_MAP[ticker] = sector
        _SECTOR_TICKERS.setdefault(_TICKER_SECTOR_MAP[ticker], []).append(ticker)


def _get_sector(ticker: str) -> str:
    if _TICKER_SECTOR_MAP is None:
        _load_sector_map()
    return (_TICKER_SECTOR_MAP or {}).get(ticker, 'UNKNOWN')


def _sector_similarity(ticker_a: str, ticker_b: str) -> float:
    sector_a = _get_sector(ticker_a)
    sector_b = _get_sector(ticker_b)
    if sector_a == 'UNKNOWN' or sector_b == 'UNKNOWN':
        return 0.0
    return 1.0 if sector_a == sector_b else 0.0


def _row_similarity(row_a: dict, row_b: dict) -> dict[str, float]:
    sent_a = float(row_a.get('sentiment', 0.0) or 0.0)
    sent_b = float(row_b.get('sentiment', 0.0) or 0.0)
    fund_a = float(row_a.get('fundamental_health', 0.5) or 0.5)
    fund_b = float(row_b.get('fundamental_health', 0.5) or 0.5)
    dq_a = float(row_a.get('data_quality', 0.6) or 0.6)
    dq_b = float(row_b.get('data_quality', 0.6) or 0.6)

    sent_sim = max(0.0, 1.0 - min(abs(sent_a - sent_b), 2.0) / 2.0)
    fund_sim = max(0.0, 1.0 - min(abs(fund_a - fund_b), 1.0))
    dq_sim = max(0.0, 1.0 - min(abs(dq_a - dq_b), 1.0))
    return {
        'sentiment_similarity': round(float(sent_sim), 4),
        'fundamental_similarity': round(float(fund_sim), 4),
        'data_quality_similarity': round(float(dq_sim), 4),
    }


def _build_explainable_relations(
    x: torch.Tensor,
    target_idx: int,
    top_k: int = 10,
) -> list[dict]:
    """Rank the most explainable related stocks for the selected ticker."""
    all_relations = _build_explainable_relations_map(x=x, top_k=top_k)
    idx_to_ticker = {idx: ticker for ticker, idx in _TICKER_IDX.items()}
    target_ticker = idx_to_ticker.get(target_idx, '')
    return list(all_relations.get(target_ticker, []))


def _build_explainable_relations_map(
    x: torch.Tensor,
    top_k: int = 10,
) -> dict[str, list[dict]]:
    """Build explainable top-k relations for all tickers in one pass."""
    idx_to_ticker = {idx: ticker for ticker, idx in _TICKER_IDX.items()}
    n_nodes = len(idx_to_ticker)
    if n_nodes <= 1:
        return {}

    corr_matrix = None
    if _ADJ_MATRIX is not None and int(_ADJ_MATRIX.shape[0]) >= n_nodes:
        corr_matrix = np.asarray(_ADJ_MATRIX, dtype=np.float32)

    # Store undirected edge attention so pair (i, j) can be queried from either side.
    attn_by_pair: dict[tuple[int, int], float] = {}
    try:
        with torch.no_grad():
            attention_table = _MODEL_CACHE.get_attention_edge_table(x, _EDGE_INDEX, _EDGE_WEIGHT)
        src_nodes = attention_table['source'].cpu().numpy()
        tgt_nodes = attention_table['target'].cpu().numpy()
        attn_mean = attention_table['attention'].mean(dim=-1).cpu().numpy()

        for edge_idx, src_idx in enumerate(src_nodes):
            src = int(src_idx)
            tgt = int(tgt_nodes[edge_idx])
            if src == tgt:
                continue
            pair = (src, tgt) if src < tgt else (tgt, src)
            attn_by_pair[pair] = max(attn_by_pair.get(pair, 0.0), float(attn_mean[edge_idx]))
    except Exception:
        attn_by_pair = {}

    profiles: list[dict] = []
    for idx in range(n_nodes):
        ticker = idx_to_ticker.get(idx, f'node_{idx}')
        profiles.append({
            'ticker': ticker,
            'sector': _get_sector(ticker),
            'sentiment': float(x[idx, 4].item()),
            'fundamental_health': float(x[idx, 5].item()),
            'data_quality': float(x[idx, 6].item()),
        })

    relation_map: dict[str, list[dict]] = {}
    for target_idx in range(n_nodes):
        target_profile = profiles[target_idx]
        target_sector = target_profile['sector']

        rows: list[dict] = []
        for idx in range(n_nodes):
            if idx == target_idx:
                continue

            profile = profiles[idx]
            ticker = profile['ticker']

            corr = 0.0
            if corr_matrix is not None:
                raw_corr = float(corr_matrix[target_idx, idx])
                corr = raw_corr if np.isfinite(raw_corr) else 0.0

            pair = (target_idx, idx) if target_idx < idx else (idx, target_idx)
            attn = float(attn_by_pair.get(pair, 0.0))
            sector_match = 1.0 if profile['sector'] == target_sector and target_sector != 'UNKNOWN' else 0.0

            similarities = _row_similarity(target_profile, profile)
            relation_score = (
                0.28 * abs(corr)
                + 0.28 * attn
                + 0.20 * sector_match
                + 0.10 * similarities['fundamental_similarity']
                + 0.08 * similarities['sentiment_similarity']
                + 0.06 * similarities['data_quality_similarity']
            )

            rows.append({
                'ticker': ticker,
                'sector': profile['sector'],
                'relation_score': round(float(relation_score), 4),
                'correlation': round(float(corr), 4),
                'attention': round(float(attn), 4),
                'sector_match': bool(sector_match),
                'same_sector': bool(sector_match),
                'sentiment_similarity': similarities['sentiment_similarity'],
                'fundamental_similarity': similarities['fundamental_similarity'],
                'data_quality_similarity': similarities['data_quality_similarity'],
            })

        rows.sort(key=lambda row: (row['relation_score'], row['sector_match'], abs(row['correlation'])), reverse=True)

        top_rows: list[dict] = []
        for rank, row in enumerate(rows[:top_k], start=1):
            ranked = dict(row)
            ranked['rank'] = rank
            top_rows.append(ranked)

        relation_map[target_profile['ticker']] = top_rows

    return relation_map


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if abs(max_value - min_value) < 1e-12:
        return [0.5 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def _fibonacci_sphere(count: int, radius: float, phase: float = 0.0) -> list[tuple[float, float, float]]:
    if count <= 0:
        return []

    points: list[tuple[float, float, float]] = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        y = 1.0 - (2.0 * (index + 0.5) / count)
        radial = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * index + phase
        x = math.cos(theta) * radial * radius
        z = math.sin(theta) * radial * radius
        points.append((x, y * radius, z))
    return points


def _load_graph_assets(device: str = 'cpu'):
    """Load model and graph structure once, cache in module globals."""
    global _MODEL_CACHE, _EDGE_INDEX, _EDGE_WEIGHT, _TICKER_IDX, _CLASS_PRIORS, _ADJ_MATRIX

    if _MODEL_CACHE is not None:
        return  # already loaded

    # Validate required files early with clear messages.
    required_graph_files = [
        GRAPH_DIR / 'edge_index.npy',
        GRAPH_DIR / 'edge_weight.npy',
        GRAPH_DIR / 'ticker_index.json',
        GRAPH_DIR / 'graph_metadata.json',
    ]

    if not GAT_CKPT.exists():
        raise FileNotFoundError(
            f"Missing GAT checkpoint: {GAT_CKPT}. "
            "Run notebooks/train_gat.ipynb (Cell 1 -> Cell 6) to produce models/gat_bvmt_best.ckpt."
        )

    for path in required_graph_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing graph asset: {path}. "
                "Run notebooks/build_graph.ipynb (all cells) to generate data/graph/*."
            )

    # Load GAT model
    _MODEL_CACHE = load_gat(str(GAT_CKPT), device=device)

    # Load graph tensors
    _EDGE_INDEX  = torch.tensor(
        np.load(GRAPH_DIR / 'edge_index.npy'),
        dtype=torch.long
    ).to(device)

    _EDGE_WEIGHT = torch.tensor(
        np.load(GRAPH_DIR / 'edge_weight.npy'),
        dtype=torch.float32
    ).to(device)

    adjacency_path = GRAPH_DIR / 'adjacency_matrix.npy'
    _ADJ_MATRIX = np.load(adjacency_path) if adjacency_path.exists() else None

    # Load ticker → node index mapping
    with open(GRAPH_DIR / 'ticker_index.json') as f:
        _TICKER_IDX = json.load(f)

    # Load leakage-safe TFT features for inference parity with training.
    _load_tft_feature_cache()
    _load_sector_map()

    # Optional class priors can be written into graph metadata by training.
    # If missing, use conservative defaults measured from prior runs.
    priors = DEFAULT_CLASS_PRIORS.copy()
    try:
        with open(GRAPH_DIR / 'graph_metadata.json') as f:
            meta = json.load(f)
        meta_priors = meta.get('class_priors')
        if isinstance(meta_priors, dict):
            for key in ('DOWN', 'FLAT', 'UP'):
                val = float(meta_priors.get(key, priors[key]))
                priors[key] = max(1e-6, min(0.999999, val))
    except Exception:
        pass

    prior_vec = torch.tensor([
        priors['DOWN'],
        priors['FLAT'],
        priors['UP'],
    ], dtype=torch.float32)
    prior_vec = prior_vec / prior_vec.sum()
    _CLASS_PRIORS = prior_vec.to(device)


class GraphAgent(BaseAgent):
    """
    Runs the trained GATBVMT on the full BVMT graph to produce
    both the legacy direction signal and a visualization-ready
    relationship graph centered on the selected stock.

    The score incorporates:
      - The stock's own technical, sentiment, fundamental signals
      - Weighted influence from correlated neighbouring stocks
      - GAT attention weights (learned during training)

    OUTPUT in state.agent_outputs['graph']:
      graph_score:    float, tanh-squashed to (-1, +1)
      graph_direction: 'UP' | 'DOWN' | 'FLAT'
      graph_class_probs: {'DOWN': p, 'FLAT': p, 'UP': p}
      top_influencers: list of (ticker, attention_weight) pairs
      confidence:     float (0 to 1)
      node_index:     int (index of target stock in graph)
    graph_view_3d:   relationship graph payload for Plotly/HTML UI
    """

    def __init__(self):
        super().__init__("graph")

    async def run(self, state: AgentState) -> AgentState:
        state.log("GraphAgent started")

        try:
            _load_graph_assets(device='cpu')

            # Check target ticker is in graph
            if state.ticker not in _TICKER_IDX:
                state.log(f"  {state.ticker} not in graph — skipping")
                state.agent_outputs['graph']     = _empty_result()
                state.confidence_scores['graph'] = 0.0
                return state

            target_idx = _TICKER_IDX[state.ticker]

            # Build full node feature matrix (N, 7) for all stocks.
            # GAT requires all nodes because attention is computed over neighbors.
            x = await self._build_node_features(state)

            # Run GAT forward pass
            _MODEL_CACHE.eval()
            with torch.no_grad():
                logits = _MODEL_CACHE(x, _EDGE_INDEX, _EDGE_WEIGHT)
                # logits shape: (N, 3)

            # Extract target stock output
            target_logits = logits[target_idx]            # (3,)
            raw_probs     = torch.softmax(target_logits, dim=-1)
            raw_pred_class = int(raw_probs.argmax().item())
            raw_direction = IDX_TO_LABEL[raw_pred_class]

            # Rebalance class tendency by down-weighting dominant priors.
            # This is the inference-time counterpart of class-weighted loss.
            adjusted_logits = target_logits - LOGIT_ADJUST_TAU * torch.log(_CLASS_PRIORS)
            probs = torch.softmax(adjusted_logits, dim=-1)
            pred_class    = int(probs.argmax().item())
            direction     = IDX_TO_LABEL[pred_class]

            # Continuous score from logits using class probabilities.
            # +1 means strongly UP, -1 means strongly DOWN.
            graph_score = float(
                torch.tanh(target_logits[2] - target_logits[0]).item()
            )

            class_probs = {
                'DOWN': round(float(probs[0].item()), 4),
                'FLAT': round(float(probs[1].item()), 4),
                'UP':   round(float(probs[2].item()), 4),
            }

            raw_class_probs = {
                'DOWN': round(float(raw_probs[0].item()), 4),
                'FLAT': round(float(raw_probs[1].item()), 4),
                'UP':   round(float(raw_probs[2].item()), 4),
            }

            # Get the top attention influencers and the top explainable relations.
            top_influencers = self._get_top_influencers(
                x, target_idx, top_k=10)
            related_stocks_top10_by_ticker = _build_explainable_relations_map(
                x=x,
                top_k=10,
            )
            related_stocks_top10 = list(related_stocks_top10_by_ticker.get(state.ticker, []))

            # Confidence: max(class_probs) rescaled to (0, 1)
            # Pure random = 0.333 → confidence 0
            # Certain = 1.0 → confidence 1.0
            max_prob   = float(probs.max().item())
            confidence = round(max(0.0, (max_prob - 0.333) / 0.667), 3)

            # Generate both ego-network and full-universe payloads
            graph_view_3d = self._build_relationship_graph(
                state=state,
                x=x,
                target_idx=target_idx,
                target_ticker=state.ticker,
                direction=direction,
                raw_direction=raw_direction,
                graph_score=graph_score,
                confidence=confidence,
                graph_class_probs=class_probs,
                raw_class_probs=raw_class_probs,
                top_influencers=top_influencers,
                related_stocks_top10=related_stocks_top10,
            )
            
            # NEW: Also generate full-universe graph for Phase 3
            full_universe_view = self._build_full_universe_graph(
                x=x,
                target_idx=target_idx,
                target_ticker=state.ticker,
                direction=direction,
                graph_score=graph_score,
                confidence=confidence,
                top_influencers=top_influencers,
                related_stocks_top10=related_stocks_top10,
                related_stocks_top10_by_ticker=related_stocks_top10_by_ticker,
            )

            result = {
                'graph_score':       round(graph_score, 4),
                'graph_direction':   direction,
                'graph_raw_direction': raw_direction,
                'graph_class_probs': class_probs,
                'graph_raw_class_probs': raw_class_probs,
                'top_influencers':   top_influencers,
                'related_stocks_top10': related_stocks_top10,
                'related_stocks_top10_by_ticker': related_stocks_top10_by_ticker,
                'confidence':        confidence,
                'node_index':        target_idx,
                'class_priors': {
                    'DOWN': round(float(_CLASS_PRIORS[0].item()), 4),
                    'FLAT': round(float(_CLASS_PRIORS[1].item()), 4),
                    'UP': round(float(_CLASS_PRIORS[2].item()), 4),
                },
                'logit_adjust_tau': LOGIT_ADJUST_TAU,
                'graph_view_3d': graph_view_3d,
                'full_universe_view': full_universe_view,  # NEW: Phase 3 full-universe
                'graph_kind': 'relationship_graph_3d_with_universe',
            }

            state.agent_outputs['graph']     = result
            state.confidence_scores['graph'] = confidence

            state.log(f"  Score={graph_score:+.4f}  "
                      f"Direction={direction}  Conf={confidence:.3f}")
            if top_influencers:
                state.log(f"  Top influencer: "
                          f"{top_influencers[0][0]} "
                          f"(attn={top_influencers[0][1]:.3f})")

        except Exception as e:
            state.log(f"  GraphAgent error: {e}")
            state.agent_outputs['graph']     = _empty_result()
            state.confidence_scores['graph'] = 0.0

        state.log("GraphAgent done")
        return state

    def _build_relationship_graph(self,
                                   state: AgentState,
                                   x: torch.Tensor,
                                   target_idx: int,
                                   target_ticker: str,
                                   direction: str,
                                   raw_direction: str,
                                   graph_score: float,
                                   confidence: float,
                                   graph_class_probs: dict,
                                   raw_class_probs: dict,
                                   top_influencers: list,
                                   related_stocks_top10: list | None = None) -> dict:
        """Build a 3D-ready graph payload centered on the target ticker."""
        with torch.no_grad():
            attention_table = _MODEL_CACHE.get_attention_edge_table(
                x, _EDGE_INDEX, _EDGE_WEIGHT)

        src_nodes = attention_table['source'].cpu().numpy()
        tgt_nodes = attention_table['target'].cpu().numpy()
        attn_mean = attention_table['attention'].mean(dim=-1).cpu().numpy()
        edge_weight = _EDGE_WEIGHT.detach().cpu().numpy()

        idx_to_ticker = {idx: ticker for ticker, idx in _TICKER_IDX.items()}

        # Build center neighborhood using both incoming and outgoing edges.
        # Some graph assets store only one edge direction, so only checking
        # incoming edges can collapse the graph into a single center node.
        center_edge_rows: list[dict] = []
        for edge_index, src_idx in enumerate(src_nodes):
            src_idx = int(src_idx)
            tgt_idx = int(tgt_nodes[edge_index])
            if src_idx == tgt_idx:
                continue
            if src_idx != target_idx and tgt_idx != target_idx:
                continue

            neighbor_idx = tgt_idx if src_idx == target_idx else src_idx
            if neighbor_idx == target_idx:
                continue

            ticker = idx_to_ticker.get(neighbor_idx, f'node_{neighbor_idx}')
            corr = float(edge_weight[edge_index])
            attn = float(attn_mean[edge_index])
            strength = 0.60 * abs(corr) + 0.40 * attn
            center_edge_rows.append({
                'neighbor_idx': neighbor_idx,
                'ticker': ticker,
                'correlation': corr,
                'attention': attn,
                'strength': strength,
                'direction': 'outgoing' if src_idx == target_idx else 'incoming',
            })

        # Aggregate multiple directed occurrences per neighbor into one row.
        neighbor_map: dict[int, dict] = {}
        for row in center_edge_rows:
            neighbor_idx = int(row['neighbor_idx'])
            entry = neighbor_map.setdefault(neighbor_idx, {
                'src_idx': neighbor_idx,
                'tgt_idx': target_idx,
                'ticker': row['ticker'],
                'correlation_sum': 0.0,
                'attention_sum': 0.0,
                'count': 0,
                'strength': 0.0,
                'directions': set(),
            })
            entry['correlation_sum'] += float(row['correlation'])
            entry['attention_sum'] += float(row['attention'])
            entry['count'] += 1
            entry['strength'] = max(float(entry['strength']), float(row['strength']))
            entry['directions'].add(str(row['direction']))

        incoming_rows: list[dict] = []
        for entry in neighbor_map.values():
            count = max(1, int(entry['count']))
            incoming_rows.append({
                'src_idx': int(entry['src_idx']),
                'tgt_idx': int(entry['tgt_idx']),
                'ticker': entry['ticker'],
                'correlation': float(entry['correlation_sum']) / count,
                'attention': float(entry['attention_sum']) / count,
                'strength': float(entry['strength']),
                'directions': sorted(entry['directions']),
            })

        max_primary_neighbors = 12
        max_total_nodes = 20
        min_context_strength = 0.18

        incoming_rows.sort(key=lambda row: row['strength'], reverse=True)
        has_direct_center_edges = len(incoming_rows) > 0
        using_adjacency_fallback = False

        # If the center node is isolated in edge_index, fall back to the
        # full adjacency matrix to keep a meaningful analysis-first graph.
        if not incoming_rows and _ADJ_MATRIX is not None and target_idx < int(_ADJ_MATRIX.shape[0]):
            center_row = np.asarray(_ADJ_MATRIX[target_idx], dtype=np.float32)
            fallback_rows: list[dict] = []
            for neighbor_idx, corr in enumerate(center_row):
                if neighbor_idx == target_idx:
                    continue
                corr = float(corr)
                if not np.isfinite(corr) or abs(corr) < 1e-8:
                    continue
                fallback_rows.append({
                    'src_idx': int(neighbor_idx),
                    'tgt_idx': int(target_idx),
                    'ticker': idx_to_ticker.get(int(neighbor_idx), f'node_{int(neighbor_idx)}'),
                    'correlation': corr,
                    'attention': 0.0,
                    'strength': abs(corr),
                    'directions': ['adjacency_fallback'],
                })

            fallback_rows.sort(key=lambda row: row['strength'], reverse=True)
            if fallback_rows:
                incoming_rows = fallback_rows
                using_adjacency_fallback = True

        selected_nodes: list[int] = [target_idx]
        primary_neighbors = incoming_rows[:max_primary_neighbors]
        for row in primary_neighbors:
            if row['src_idx'] not in selected_nodes:
                selected_nodes.append(row['src_idx'])

        # Expand the subgraph with the strongest 2-hop context while keeping
        # the visualization readable.
        selected_set = set(selected_nodes)
        for edge_index, src_idx in enumerate(src_nodes):
            tgt_idx = int(tgt_nodes[edge_index])
            if src_idx == tgt_idx:
                continue

            correlation = float(edge_weight[edge_index])
            attention = float(attn_mean[edge_index])
            strength = 0.55 * abs(correlation) + 0.45 * attention

            if src_idx in selected_set or tgt_idx in selected_set:
                if len(selected_nodes) < max_total_nodes:
                    other_idx = int(tgt_idx) if int(src_idx) in selected_set else int(src_idx)
                    if other_idx not in selected_set and strength >= min_context_strength:
                        selected_nodes.append(other_idx)
                        selected_set.add(other_idx)

        # Recompute using the final selected set so the output graph is stable.
        selected_set = set(selected_nodes)
        pair_map: dict[tuple[int, int], dict] = {}
        for edge_index, src_idx in enumerate(src_nodes):
            tgt_idx = int(tgt_nodes[edge_index])
            if src_idx == tgt_idx:
                continue
            if int(src_idx) not in selected_set or tgt_idx not in selected_set:
                continue

            pair = tuple(sorted((int(src_idx), tgt_idx)))
            corr = float(edge_weight[edge_index])
            attn = float(attn_mean[edge_index])
            entry = pair_map.setdefault(pair, {
                'source_idx': pair[0],
                'target_idx': pair[1],
                'correlations': [],
                'attentions': [],
                'directed': [],
            })
            entry['correlations'].append(corr)
            entry['attentions'].append(attn)
            entry['directed'].append({
                'source': idx_to_ticker.get(int(src_idx), f'node_{int(src_idx)}'),
                'target': idx_to_ticker.get(tgt_idx, f'node_{tgt_idx}'),
                'correlation': round(corr, 4),
                'attention': round(attn, 4),
            })

        edges_raw: list[dict] = []
        for pair, entry in pair_map.items():
            corr_avg = float(np.mean(entry['correlations']))
            attn_avg = float(np.mean(entry['attentions']))
            combined = 0.55 * abs(corr_avg) + 0.45 * attn_avg
            edges_raw.append({
                'source_idx': entry['source_idx'],
                'target_idx': entry['target_idx'],
                'correlation': round(corr_avg, 4),
                'attention': round(attn_avg, 4),
                'weight_raw': combined,
                'directed': entry['directed'],
            })

        # Ensure center-to-neighbor links exist when operating in adjacency
        # fallback mode (center node had no direct edges in edge_index).
        if using_adjacency_fallback:
            existing_center_neighbors: set[int] = set()
            for edge in edges_raw:
                source_idx = int(edge['source_idx'])
                target_idx_edge = int(edge['target_idx'])
                if source_idx == target_idx:
                    existing_center_neighbors.add(target_idx_edge)
                elif target_idx_edge == target_idx:
                    existing_center_neighbors.add(source_idx)

            for row in primary_neighbors:
                neighbor_idx = int(row['src_idx'])
                if neighbor_idx in existing_center_neighbors:
                    continue

                corr = float(row['correlation'])
                edges_raw.append({
                    'source_idx': int(target_idx),
                    'target_idx': neighbor_idx,
                    'correlation': round(corr, 4),
                    'attention': 0.0,
                    'weight_raw': abs(corr),
                    'directed': [
                        {
                            'source': target_ticker,
                            'target': row['ticker'],
                            'correlation': round(corr, 4),
                            'attention': 0.0,
                            'mode': 'adjacency_fallback',
                        }
                    ],
                })

        edge_weights = [edge['weight_raw'] for edge in edges_raw]
        edge_weight_norm = _normalize(edge_weights)
        for edge, norm_weight in zip(edges_raw, edge_weight_norm):
            edge['weight'] = round(float(norm_weight), 4)
            edge['width'] = round(0.9 + 4.1 * float(norm_weight), 3)
            edge['opacity'] = round(0.12 + 0.78 * float(norm_weight), 3)
            edge['color'] = '#22c55e' if edge['correlation'] >= 0 else '#ef4444'
            if using_adjacency_fallback and edge['attention'] == 0.0:
                edge['kind'] = 'adjacency_fallback_link'
            else:
                edge['kind'] = 'center_link' if target_idx in (edge['source_idx'], edge['target_idx']) else 'context_link'

        adjacency_strength: dict[int, float] = {node_idx: 0.0 for node_idx in selected_nodes}
        adjacency_attention: dict[int, float] = {node_idx: 0.0 for node_idx in selected_nodes}
        adjacency_degree: dict[int, int] = {node_idx: 0 for node_idx in selected_nodes}
        for edge in edges_raw:
            source_idx = edge['source_idx']
            target_idx_edge = edge['target_idx']
            weight = float(edge['weight'])
            attention = float(edge['attention'])
            adjacency_strength[source_idx] = adjacency_strength.get(source_idx, 0.0) + weight
            adjacency_strength[target_idx_edge] = adjacency_strength.get(target_idx_edge, 0.0) + weight
            adjacency_attention[source_idx] = adjacency_attention.get(source_idx, 0.0) + attention
            adjacency_attention[target_idx_edge] = adjacency_attention.get(target_idx_edge, 0.0) + attention
            adjacency_degree[source_idx] = adjacency_degree.get(source_idx, 0) + 1
            adjacency_degree[target_idx_edge] = adjacency_degree.get(target_idx_edge, 0) + 1

        node_strengths = [adjacency_strength[node_idx] for node_idx in selected_nodes]
        node_strength_norm = _normalize(node_strengths)
        phase = (float(_stable_seed(target_ticker)) % 360.0) / 180.0 * math.pi
        primary_points = _fibonacci_sphere(len(primary_neighbors), 1.9, phase=phase)
        context_nodes = [node_idx for node_idx in selected_nodes if node_idx not in [target_idx] + [row['src_idx'] for row in primary_neighbors]]
        context_points = _fibonacci_sphere(len(context_nodes), 3.2, phase=phase + 0.9)

        position_map: dict[int, tuple[float, float, float]] = {target_idx: (0.0, 0.0, 0.0)}
        for row, point in zip(primary_neighbors, primary_points):
            position_map[row['src_idx']] = point
        for node_idx, point in zip(context_nodes, context_points):
            position_map[node_idx] = point

        # Stable fallback in case a node was selected through edges but not assigned yet.
        remaining_nodes = [node_idx for node_idx in selected_nodes if node_idx not in position_map]
        remaining_points = _fibonacci_sphere(len(remaining_nodes), 4.1, phase=phase + 1.7)
        for node_idx, point in zip(remaining_nodes, remaining_points):
            position_map[node_idx] = point

        node_roles = {target_idx: 'center'}
        for row in primary_neighbors:
            node_roles[row['src_idx']] = 'influencer'
        for node_idx in context_nodes:
            node_roles.setdefault(node_idx, 'related')
        for node_idx in selected_nodes:
            node_roles.setdefault(node_idx, 'context')

        node_palette = {
            'center': '#ff4d4d',
            'influencer': '#ffb020',
            'related': '#5aa9ff',
            'context': '#a0aec0',
        }

        nodes: list[dict] = []
        sorted_nodes = [target_idx] + [row['src_idx'] for row in primary_neighbors] + context_nodes
        seen_nodes: set[int] = set()
        for node_idx in sorted_nodes:
            if node_idx in seen_nodes:
                continue
            seen_nodes.add(node_idx)

            ticker = idx_to_ticker.get(int(node_idx), f'node_{int(node_idx)}')
            role = node_roles.get(node_idx, 'context')
            position = position_map.get(node_idx, (0.0, 0.0, 0.0))
            strength = float(adjacency_strength.get(node_idx, 0.0))
            strength_index = node_strength_norm[selected_nodes.index(node_idx)] if node_idx in selected_nodes else 0.5
            node_attention = float(adjacency_attention.get(node_idx, 0.0))
            node_degree = int(adjacency_degree.get(node_idx, 0))

            if node_idx == target_idx:
                size = 28.0
                color = node_palette['center']
            else:
                size = 10.0 + 16.0 * strength_index
                color = node_palette.get(role, '#a0aec0')

            nodes.append({
                'ticker': ticker,
                'label': ticker,
                'node_index': int(node_idx),
                'sector': _get_sector(ticker),
                'x': round(float(position[0]), 4),
                'y': round(float(position[1]), 4),
                'z': round(float(position[2]), 4),
                'size': round(float(size), 3),
                'color': color,
                'role': role,
                'degree': node_degree,
                'centrality': round(_clamp(strength_index, 0.0, 1.0), 4),
                'attention': round(float(node_attention), 4),
                'strength': round(float(strength), 4),
                'confidence': round(float(confidence), 4) if node_idx == target_idx else None,
            })

        strongest_connection = None
        if edges_raw:
            strongest_edge = max(edges_raw, key=lambda edge: edge['weight'])
            strongest_connection = {
                'source': idx_to_ticker.get(int(strongest_edge['source_idx']), f'node_{int(strongest_edge["source_idx"])}'),
                'target': idx_to_ticker.get(int(strongest_edge['target_idx']), f'node_{int(strongest_edge["target_idx"])}'),
                'correlation': strongest_edge['correlation'],
                'attention': strongest_edge['attention'],
                'weight': strongest_edge['weight'],
            }

        top_influencer_details = []
        for ticker, attn in top_influencers:
            top_influencer_details.append({
                'ticker': ticker,
                'attention': attn,
            })

        expansion_candidates = [
            {
                'ticker': row['ticker'],
                'node_index': int(row['src_idx']),
                'attention': round(float(row['attention']), 4),
                'correlation': round(float(row['correlation']), 4),
                'strength': round(float(row['strength']), 4),
            }
            for row in incoming_rows[max_primary_neighbors:]
        ]

        return {
            'center_ticker': target_ticker,
            'layout_mode': '3d_relationship',
            'nodes': nodes,
            'edges': [
                {
                    'source': idx_to_ticker.get(int(edge['source_idx']), f"node_{int(edge['source_idx'])}"),
                    'target': idx_to_ticker.get(int(edge['target_idx']), f"node_{int(edge['target_idx'])}"),
                    'source_idx': int(edge['source_idx']),
                    'target_idx': int(edge['target_idx']),
                    'correlation': edge['correlation'],
                    'attention': edge['attention'],
                    'weight': edge['weight'],
                    'width': edge['width'],
                    'opacity': edge['opacity'],
                    'color': edge['color'],
                    'kind': edge['kind'],
                    'directed': edge['directed'],
                }
                for edge in edges_raw
            ],
            'summary': {
                'top_influencers': top_influencers,
                'top_influencers_detail': top_influencer_details,
                'related_stocks_top10': related_stocks_top10 or [],
                'number_of_nodes': len(nodes),
                'number_of_edges': len(edges_raw),
                'strongest_connection': strongest_connection,
                'graph_confidence': round(float(confidence), 4),
                'graph_score': round(float(graph_score), 4),
                'graph_direction': direction,
                'graph_raw_direction': raw_direction,
                'graph_class_probs': graph_class_probs,
                'graph_raw_class_probs': raw_class_probs,
                'max_primary_neighbors': max_primary_neighbors,
                'expansion_candidates': expansion_candidates,
                'center_neighbor_count': len(incoming_rows),
                'has_center_edges': has_direct_center_edges,
                'used_adjacency_fallback': using_adjacency_fallback,
                'note': (
                    None if has_direct_center_edges else (
                        'Selected ticker has no direct graph edges; using adjacency correlations as fallback.'
                        if using_adjacency_fallback
                        else 'Selected ticker has no direct graph edges in current graph assets.'
                    )
                ),
            },
        }

    async def _build_node_features(self,
                                   state: AgentState) -> torch.Tensor:
        """
        Build (N, 7) feature matrix for all stocks.
        Core technical features (0..3) come from leakage-safe TFT columns
        to keep training and inference aligned.
        Sentiment/fundamental/data-quality features are enriched from DB
        and state outputs when available.
        """
        n_stocks = len(_TICKER_IDX)
        x        = torch.zeros(n_stocks, 7, dtype=torch.float32)

        # Defaults for robustness if a source is missing.
        x[:, 0] = 0.5
        x[:, 1] = 0.5
        x[:, 2] = 0.0
        x[:, 3] = 0.0
        x[:, 4] = 0.0
        x[:, 5] = 0.5
        x[:, 6] = 0.6

        # Ensure TFT cache exists (already loaded in _load_graph_assets normally).
        if _TFT_LATEST_FEATURES is None or _TFT_DQ_SCORES is None:
            _load_tft_feature_cache()

        # Fill technical features from latest leakage-safe TFT row per ticker.
        for ticker, idx in _TICKER_IDX.items():
            feat = (_TFT_LATEST_FEATURES or {}).get(ticker, {})

            rsi = feat.get('rsi_14', np.nan)
            vol = feat.get('volatility_20', np.nan)
            mar = feat.get('price_to_ma20', np.nan)
            ret = feat.get('daily_return', np.nan)

            if pd.notna(rsi):
                x[idx, 0] = _clamp(float(rsi) / 100.0, 0.0, 1.0)
            if pd.notna(vol):
                x[idx, 1] = _sigmoid_norm(float(vol), scale=10.0)
            if pd.notna(mar):
                x[idx, 2] = _clamp(float(mar) / 0.2, -1.0, 1.0)
            if pd.notna(ret):
                x[idx, 3] = _clamp(float(ret) / 0.1, -1.0, 1.0)

            dq = (_TFT_DQ_SCORES or {}).get(ticker, 0.6)
            x[idx, 6] = _clamp(dq, 0.0, 1.0)

        # Fetch latest sentiment for all stocks
        sent_rows = await self.query_db('''
            SELECT DISTINCT ON (ticker)
                ticker, sentiment_score
            FROM news_articles
            WHERE sentiment_label IS NOT NULL
            ORDER BY ticker, published_at DESC
        ''')

        for row in sent_rows:
            idx = _TICKER_IDX.get(row['ticker'])
            if idx is not None:
                x[idx, 4] = max(-1.0, min(1.0,
                                 float(row['sentiment_score'] or 0.0)))

        # Fill fundamentals for all stocks from latest extracted ratios.
        fund_rows = await self.query_db('''
            SELECT DISTINCT ON (ticker)
                ticker, company_type, roe, roa, cost_income, extraction_confidence
            FROM financial_ratios
            WHERE extraction_confidence >= 0.5
            ORDER BY ticker, period_end_date DESC
        ''')

        for row in fund_rows:
            idx = _TICKER_IDX.get(row['ticker'])
            if idx is None:
                continue
            score = _fund_score_from_fields(
                row.get('company_type'),
                row.get('roe'),
                row.get('roa'),
                row.get('cost_income'),
            )
            x[idx, 5] = _clamp(score, 0.0, 1.0)

        # Override target stock features with richer agent outputs.
        # NOTE: keep feature indexes aligned with training notebook.
        target_idx = _TICKER_IDX[state.ticker]

        # Sentiment from SentimentAgent
        sent = state.agent_outputs.get('sentiment', {})
        x[target_idx, 4] = max(-1.0, min(1.0,
                                float(sent.get('sentiment_signal', 0.0))))

        # Fundamental health (robust to both new and old field names)
        fund = state.agent_outputs.get('fundamental', {})
        health_map = {'CRITICAL': 0.0, 'WEAK': 0.33,
                      'MODERATE': 0.67, 'STRONG': 1.0}

        if 'normalized_health_score' in fund:
            fund_value = float(fund.get('normalized_health_score', 0.5))
        elif 'health_label' in fund:
            fund_value = health_map.get(str(fund.get('health_label', 'MODERATE')).upper(), 0.5)
        elif 'health_score' in fund:
            fund_value = float(fund.get('health_score', 1.5)) / 3.0
        else:
            fund_value = 0.5

        x[target_idx, 5] = max(0.0, min(1.0, fund_value))

        # Data quality from DataAgent, with TFT coverage as fallback.
        target_dq = (_TFT_DQ_SCORES or {}).get(state.ticker, 0.6)
        if hasattr(state, 'data_quality_score'):
            target_dq = float(state.data_quality_score)
        x[target_idx, 6] = _clamp(target_dq, 0.0, 1.0)

        return x

    def _build_full_universe_graph(self,
                                    x: torch.Tensor,
                                    target_idx: int,
                                    target_ticker: str,
                                    direction: str,
                                    graph_score: float,
                                    confidence: float,
                                    top_influencers: list,
                                    related_stocks_top10: list | None = None,
                                    related_stocks_top10_by_ticker: dict[str, list[dict]] | None = None) -> dict:
        """
        Build a full-universe graph payload with ALL stocks visible.
        
        This is the Phase 3 redesign: instead of ego-network, we return all nodes
        and all edges with metadata for dynamic per-ticker focus.
        
        The HTML renderer will use this to:
        1. Show all nodes initially
        2. Allow smooth camera animation to any selected ticker
        3. Highlight selected node and its relations
        4. Provide per-ticker interactive controls
        """
        with torch.no_grad():
            attention_table = _MODEL_CACHE.get_attention_edge_table(
                x, _EDGE_INDEX, _EDGE_WEIGHT)

        src_nodes = attention_table['source'].cpu().numpy()
        tgt_nodes = attention_table['target'].cpu().numpy()
        attn_mean = attention_table['attention'].mean(dim=-1).cpu().numpy()
        edge_weight = _EDGE_WEIGHT.detach().cpu().numpy()

        idx_to_ticker = {idx: ticker for ticker, idx in _TICKER_IDX.items()}
        n_stocks = len(_TICKER_IDX)

        # Build complete edge map for all pairs
        pair_map: dict[tuple[int, int], dict] = {}
        for edge_index, src_idx in enumerate(src_nodes):
            src_idx = int(src_idx)
            tgt_idx = int(tgt_nodes[edge_index])
            if src_idx == tgt_idx:
                continue

            pair = tuple(sorted((src_idx, tgt_idx)))
            corr = float(edge_weight[edge_index])
            attn = float(attn_mean[edge_index])
            entry = pair_map.setdefault(pair, {
                'source_idx': pair[0],
                'target_idx': pair[1],
                'correlations': [],
                'attentions': [],
                'directed': [],
            })
            entry['correlations'].append(corr)
            entry['attentions'].append(attn)
            entry['directed'].append({
                'source': idx_to_ticker.get(int(src_idx), f'node_{int(src_idx)}'),
                'target': idx_to_ticker.get(int(tgt_idx), f'node_{int(tgt_idx)}'),
                'correlation': round(corr, 4),
                'attention': round(attn, 4),
            })

        # Build edge list with normalized weights
        edges_raw: list[dict] = []
        for pair, entry in pair_map.items():
            corr_avg = float(np.mean(entry['correlations']))
            attn_avg = float(np.mean(entry['attentions']))
            combined = 0.55 * abs(corr_avg) + 0.45 * attn_avg
            edges_raw.append({
                'source_idx': entry['source_idx'],
                'target_idx': entry['target_idx'],
                'correlation': round(corr_avg, 4),
                'attention': round(attn_avg, 4),
                'weight_raw': combined,
                'directed': entry['directed'],
            })

        edge_weights = [edge['weight_raw'] for edge in edges_raw]
        edge_weight_norm = _normalize(edge_weights) if edge_weights else []
        for edge, norm_weight in zip(edges_raw, edge_weight_norm):
            edge['weight'] = round(float(norm_weight), 4)
            edge['width'] = round(0.9 + 4.1 * float(norm_weight), 3)
            edge['opacity'] = round(0.12 + 0.78 * float(norm_weight), 3)
            edge['color'] = '#22c55e' if edge['correlation'] >= 0 else '#ef4444'
            edge['kind'] = 'strong_correlation' if abs(edge['correlation']) >= 0.3 else 'weak_correlation'

        # Compute node-level stats for all nodes
        node_degree: dict[int, int] = {i: 0 for i in range(n_stocks)}
        node_strength: dict[int, float] = {i: 0.0 for i in range(n_stocks)}
        node_attention: dict[int, float] = {i: 0.0 for i in range(n_stocks)}

        for edge in edges_raw:
            src_idx = int(edge['source_idx'])
            tgt_idx = int(edge['target_idx'])
            weight = float(edge['weight'])
            attn = float(edge['attention'])

            node_degree[src_idx] = node_degree.get(src_idx, 0) + 1
            node_degree[tgt_idx] = node_degree.get(tgt_idx, 0) + 1
            node_strength[src_idx] = node_strength.get(src_idx, 0.0) + weight
            node_strength[tgt_idx] = node_strength.get(tgt_idx, 0.0) + weight
            node_attention[src_idx] = node_attention.get(src_idx, 0.0) + attn
            node_attention[tgt_idx] = node_attention.get(tgt_idx, 0.0) + attn

        # Compute global node strength normalization
        all_strengths = list(node_strength.values())
        strength_norm = _normalize(all_strengths)
        strength_map = {i: sn for i, sn in enumerate(strength_norm)}

        # Use stable positions based on ticker hash
        def node_position_3d(idx: int) -> tuple[float, float, float]:
            """Generate stable 3D position for each node based on its index."""
            # Use fibonacci sphere with ticker-based phase shift
            ticker = idx_to_ticker.get(idx, f'node_{idx}')
            phase = (float(_stable_seed(ticker)) % 360.0) / 180.0 * math.pi
            points = _fibonacci_sphere(n_stocks, 5.0, phase=phase)
            if idx < len(points):
                return points[idx]
            return (0.0, 0.0, 0.0)

        positions = {i: node_position_3d(i) for i in range(n_stocks)}
        related_lookup = {row['ticker']: row for row in (related_stocks_top10 or [])}

        # Build all nodes
        nodes: list[dict] = []
        for node_idx in range(n_stocks):
            ticker = idx_to_ticker.get(node_idx, f'node_{node_idx}')
            position = positions.get(node_idx, (0.0, 0.0, 0.0))
            degree = node_degree.get(node_idx, 0)
            strength = node_strength.get(node_idx, 0.0)
            strength_index = strength_map.get(node_idx, 0.5)
            attention_sum = node_attention.get(node_idx, 0.0)

            # Node size scaled by strength
            if node_idx == target_idx:
                size = 32.0
                color = '#ff4d4d'  # red for selected
                role = 'target'
            elif ticker in related_lookup:
                size = 10.5 + 17.0 * strength_index
                color = '#ffb020'
                role = 'related'
            else:
                size = 8.0 + 18.0 * strength_index
                # Color by role: top influencers are orange, others are blue-gray
                if any(t == ticker for t, _ in top_influencers):
                    color = '#ffb020'  # orange for influencers
                    role = 'influencer'
                else:
                    color = '#5aa9ff' if strength_index > 0.3 else '#a0aec0'
                    role = 'normal' if strength_index > 0.3 else 'context'

            nodes.append({
                'ticker': ticker,
                'label': ticker,
                'node_index': int(node_idx),
                'sector': _get_sector(ticker),
                'x': round(float(position[0]), 4),
                'y': round(float(position[1]), 4),
                'z': round(float(position[2]), 4),
                'size': round(float(size), 3),
                'color': color,
                'role': role,
                'degree': degree,
                'centrality': round(_clamp(strength_index, 0.0, 1.0), 4),
                'attention': round(float(attention_sum), 4),
                'strength': round(float(strength), 4),
                'is_target': node_idx == target_idx,
            })

        # Build edges with source/target tickers
        edges: list[dict] = [
            {
                'source': idx_to_ticker.get(int(edge['source_idx']), f"node_{int(edge['source_idx'])}"),
                'target': idx_to_ticker.get(int(edge['target_idx']), f"node_{int(edge['target_idx'])}"),
                'source_idx': int(edge['source_idx']),
                'target_idx': int(edge['target_idx']),
                'correlation': edge['correlation'],
                'attention': edge['attention'],
                'weight': edge['weight'],
                'width': edge['width'],
                'opacity': edge['opacity'],
                'color': edge['color'],
                'kind': edge['kind'],
            }
            for edge in edges_raw
        ]

        # Find strongest connection
        strongest_connection = None
        if edges_raw:
            strongest_edge = max(edges_raw, key=lambda edge: edge['weight'])
            strongest_connection = {
                'source': idx_to_ticker.get(int(strongest_edge['source_idx']), f"node_{int(strongest_edge['source_idx'])}"),
                'target': idx_to_ticker.get(int(strongest_edge['target_idx']), f"node_{int(strongest_edge['target_idx'])}"),
                'correlation': strongest_edge['correlation'],
                'attention': strongest_edge['attention'],
                'weight': strongest_edge['weight'],
            }

        return {
            'center_ticker': target_ticker,
            'layout_mode': 'full_universe_3d',
            'nodes': nodes,
            'edges': edges,
            'summary': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'strongest_connection': strongest_connection,
                'target_ticker': target_ticker,
                'target_direction': direction,
                'target_score': round(float(graph_score), 4),
                'target_confidence': round(float(confidence), 4),
                'top_influencers': top_influencers,
                'related_stocks_top10': related_stocks_top10 or [],
                'related_stocks_top10_by_ticker': related_stocks_top10_by_ticker or {},
                'universe_size': n_stocks,
                'note': f'Full universe view: {n_stocks} stocks, {len(edges)} correlations. Click any stock to focus.',
            },
        }

    def _get_top_influencers(self,
                              x: torch.Tensor,
                              target_idx: int,
                              top_k: int = 5) -> list:
        """
        Get the top-k stocks that most influence the target stock,
        based on GAT attention weights from layer 1.

        Returns list of (ticker, avg_attention_weight) sorted desc.
        """
        try:
            with torch.no_grad():
                alpha = _MODEL_CACHE.get_attention_weights(
                    x, _EDGE_INDEX, _EDGE_WEIGHT)
                # alpha: (E, n_heads)

            # Find edges where target is the destination
            tgt_mask = (_EDGE_INDEX[1] == target_idx)
            src_nodes = _EDGE_INDEX[0][tgt_mask].cpu().numpy()
            attn      = alpha[tgt_mask].mean(dim=-1).cpu().numpy()

            # Sort by attention weight
            order = np.argsort(attn)[::-1]

            # Reverse lookup: node_index → ticker
            idx_to_ticker = {v: k for k, v in _TICKER_IDX.items()}

            result = []
            for i in order[:top_k]:
                src_idx = int(src_nodes[i])
                if src_idx == target_idx:
                    continue   # skip self-loop
                ticker = idx_to_ticker.get(src_idx, f'node_{src_idx}')
                result.append((ticker, round(float(attn[i]), 4)))

            return result

        except Exception:
            return []


def _empty_result() -> dict:
    return {
        'graph_score':       0.0,
        'graph_direction':   'FLAT',
        'graph_class_probs': {'DOWN': 0.333, 'FLAT': 0.334, 'UP': 0.333},
        'top_influencers':   [],
        'confidence':        0.0,
        'node_index':        -1,
        'graph_view_3d': {
            'center_ticker': None,
            'layout_mode': '3d_relationship',
            'nodes': [],
            'edges': [],
            'summary': {},
        },
        'graph_kind': 'relationship_graph_3d',
    }
