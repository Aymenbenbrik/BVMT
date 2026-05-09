"""
/graph-data endpoint.

Reads correlation data directly from numpy files — no agent call at
request time. Fast and always available even before any stocks are cached.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter

from app.backend.models import (
    GraphDataResponse,
    GraphEdge,
    GraphMetadataInfo,
    GraphNode,
)

router = APIRouter(tags=["graph"])

ROOT = Path(__file__).resolve().parent.parent.parent.parent
GRAPH_DIR = ROOT / "data" / "graph"


def _get_company_type(ticker: str) -> str:
    banks = {
        "AMEN BANK", "BIAT", "BNA", "STB", "BH BANK",
        "ATTIJARI BANK", "ATB", "UBCI", "UIB", "BT", "WIFACK INT BANK",
    }
    insurance = {
        "STAR", "ASTREE", "BH ASSURANCE", "ASSUR MAGHREBIA",
        "ASSU MAGHREBIA VIE", "TUNIS RE", "BNA ASSURANCES",
    }
    if ticker in banks:
        return "bank"
    if ticker in insurance:
        return "insurance"
    return "non_bank"


def _cache():
    from app.backend.routers.prediction import cache
    return cache


@router.get("/graph-data", response_model=GraphDataResponse)
async def graph_data(threshold: float = 0.4):
    """
    Return graph nodes and edges for the 3D network visualization.
    Reads from data/graph/ npy files directly — no agent pipeline needed.
    Threshold param controls minimum edge weight shown (default 0.4).
    """
    # Load ticker index
    ticker_index_path = GRAPH_DIR / "ticker_index.json"
    with open(ticker_index_path, "r", encoding="utf-8") as f:
        ticker_index: dict = json.load(f)  # {ticker: node_idx}

    idx_to_ticker = {v: k for k, v in ticker_index.items()}
    n_nodes = len(ticker_index)

    # Load metadata
    metadata_path = GRAPH_DIR / "graph_metadata.json"
    meta = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # Load adjacency matrix for edge weights
    adj_path = GRAPH_DIR / "adjacency_matrix.npy"
    adj: Optional[np.ndarray] = None
    if adj_path.exists():
        adj = np.load(adj_path)

    # Load edge index + weight tensors for precise edge list
    edge_index_path = GRAPH_DIR / "edge_index.npy"
    edge_weight_path = GRAPH_DIR / "edge_weight.npy"

    edges_raw = []
    if edge_index_path.exists() and edge_weight_path.exists():
        edge_index = np.load(edge_index_path)  # shape (2, E)
        edge_weight = np.load(edge_weight_path)  # shape (E,)

        seen_pairs: set = set()
        count = 0
        for i in range(edge_index.shape[1]):
            src_idx = int(edge_index[0, i])
            tgt_idx = int(edge_index[1, i])
            if src_idx == tgt_idx:
                continue
            w = float(edge_weight[i])
            if abs(w) < threshold:
                continue
            pair = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges_raw.append((src_idx, tgt_idx, w))
            count += 1
            if count >= 300:  # cap at 300 edges for perf
                break

    # Pull cached direction/confidence for nodes
    c = _cache()
    cached_all = c.all_cached()

    # Build nodes
    nodes: List[GraphNode] = []
    for idx in range(n_nodes):
        ticker = idx_to_ticker.get(idx, f"node_{idx}")
        entry = cached_all.get(ticker)
        direction = None
        confidence = None
        final_score = None
        if entry is not None:
            d = entry.data
            direction = d.get("final_direction")
            confidence = d.get("overall_confidence")
            if confidence is not None:
                final_score = round(float(confidence) * 100, 1)

        nodes.append(GraphNode(
            id=ticker,
            label=ticker,
            company_type=_get_company_type(ticker),
            sector=None,
            final_direction=direction,
            final_confidence=confidence,
            final_score=final_score,
        ))

    # Build edges
    edges: List[GraphEdge] = []
    for src_idx, tgt_idx, w in edges_raw:
        src_ticker = idx_to_ticker.get(src_idx, f"node_{src_idx}")
        tgt_ticker = idx_to_ticker.get(tgt_idx, f"node_{tgt_idx}")
        abs_w = abs(w)
        corr_type = "positive" if w >= 0 else "negative"
        if abs_w >= 0.7:
            strength = "STRONG"
        elif abs_w >= 0.4:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        edges.append(GraphEdge(
            source=src_ticker,
            target=tgt_ticker,
            weight=round(abs_w, 4),
            correlation_type=corr_type,
            strength_label=strength,
        ))

    graph_info = GraphMetadataInfo(
        total_nodes=n_nodes,
        total_edges=len(edges),
        computation_date=meta.get("built_at"),
        correlation_window_days=int(meta.get("correlation_window", 120)),
    )

    return GraphDataResponse(
        nodes=nodes,
        edges=edges,
        graph_metadata=graph_info,
    )
