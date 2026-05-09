"""
On-demand LLM narrative generation for the Explanation tab.

- POST /explain/generate  — calls LLM to produce a short narrative
- GET  /explain/narrative/{ticker} — returns cached narrative if available

The LLM is NEVER called during batch/precompute. Only when a user
explicitly clicks "Generate narrative" for a single ticker.
"""

from __future__ import annotations

import os
import time
import json
import re
import httpx
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/explain", tags=["explain"])

from app.backend.routers.prediction import cache

# ── Pydantic models ─────────────────────────────────────────────────

class NarrativeRequest(BaseModel):
    ticker: str
    explanation_facts: Dict[str, Any] = Field(default_factory=dict)


class NarrativeResponse(BaseModel):
    narrative: str
    model_used: str = ""
    generated_at: str = ""
    cached: bool = False


class XaiNarrativeRequest(BaseModel):
    ticker: str
    explain_version: str = "v1"
    technical: Dict[str, Any] = Field(default_factory=dict)
    fundamental: Dict[str, Any] = Field(default_factory=dict)
    graph: Dict[str, Any] = Field(default_factory=dict)


class XaiNarrativeResponse(BaseModel):
    technical: str
    fundamental: str
    graph: str
    model_used: str = ""
    generated_at: str = ""
    cached: bool = False


# ── Helpers ──────────────────────────────────────────────────────────

def _get_cached_narrative(ticker: str, explain_version: str) -> Optional[Dict[str, Any]]:
    cache_key = f"narrative:{ticker}:{explain_version}"
    entry = cache.get(cache_key)
    if entry:
        return entry.data
    return None


def _set_cached_narrative(ticker: str, explain_version: str, narrative: str, model_used: str) -> None:
    cache_key = f"narrative:{ticker}:{explain_version}"
    data = {
        "narrative": narrative,
        "model_used": model_used,
        "generated_at": datetime.now().isoformat(),
    }
    cache.set(cache_key, data)


def _get_cached_xai_narrative(ticker: str, explain_version: str) -> Optional[Dict[str, Any]]:
    cache_key = f"xai-narrative:{ticker}:{explain_version}"
    entry = cache.get(cache_key)
    if entry:
        return entry.data
    return None


def _set_cached_xai_narrative(
    ticker: str,
    explain_version: str,
    technical: str,
    fundamental: str,
    graph: str,
    model_used: str,
) -> None:
    cache_key = f"xai-narrative:{ticker}:{explain_version}"
    data = {
        "technical": technical,
        "fundamental": fundamental,
        "graph": graph,
        "model_used": model_used,
        "generated_at": datetime.now().isoformat(),
    }
    cache.set(cache_key, data)


def _build_prompt(facts: Dict[str, Any]) -> str:
    """Build the user payload for the LLM from explanation_facts."""
    lines = []
    verdict = facts.get("verdict", "NEUTRAL")
    confidence = facts.get("confidence", 0)
    lines.append(f"Verdict: {verdict} with {confidence}% confidence.")

    for section in ["technical", "fundamental", "sentiment", "graph"]:
        sec = facts.get(section, {})
        if sec:
            summary = sec.get("summary", "")
            reasons = sec.get("reasons", [])
            if summary:
                lines.append(f"{section.title()}: {summary}")
            if reasons:
                lines.append(f"  Reasons: {', '.join(str(r) for r in reasons)}")

    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a concise, neutral financial explainer for the Tunisian stock market (BVMT). "
    "Reword the provided facts into a short, factual, non-speculative summary. "
    "Produce exactly 3 short paragraphs: (1) headline verdict, (2) cause summary, (3) what to watch. "
    "Maximum 120 words total. Do not add disclaimers or speculation."
)

XAI_SYSTEM_PROMPT = (
    "You are a concise, neutral financial explainer for the Tunisian stock market (BVMT). "
    "Use only the provided facts. Do not remove any item from the lists. "
    "Write one short paragraph per section. Mention all factors explicitly. "
    "Output format must be exactly:\n"
    "TECHNICAL: <text>\nFUNDAMENTAL: <text>\nGRAPH: <text>\n"
    "No markdown or code fences."
)


async def _call_llm_messages(
    messages: list[Dict[str, str]],
    response_format: Optional[Dict[str, str]] = None,
) -> tuple[str, str]:
    """
    Call the configured LLM. Uses OpenAI-compatible API (works with
    OpenAI, Groq, Together, local Ollama, etc.).
    """
    api_key = os.getenv("EXPLAINER_LLM_API_KEY", "")
    base_url = os.getenv("EXPLAINER_LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("EXPLAINER_LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="LLM narrative generation is not configured. Set EXPLAINER_LLM_API_KEY in your .env file.",
        )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 250,
    }
    if response_format:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Add GitHub Models compatible headers when using a GitHub-style base URL
    if any(x in base_url.lower() for x in ("github", "inference.ai", "models.inference")):
        headers.setdefault("Accept", "application/vnd.github+json")
        # Add a generic API version header; this is accepted by GitHub endpoints.
        headers.setdefault("X-GitHub-Api-Version", "2022-11-28")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="LLM request timed out")
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Cannot reach LLM API")

        if response_format and resp.status_code == 400:
            body = resp.text
            if "response_format" in body or "json" in body:
                payload.pop("response_format", None)
                try:
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                except httpx.TimeoutException:
                    raise HTTPException(status_code=504, detail="LLM request timed out")
                except httpx.ConnectError:
                    raise HTTPException(status_code=503, detail="Cannot reach LLM API")

        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail="LLM rate limit reached. Please try again later.")
        if resp.status_code != 200:
            # Surface response text for easier debugging in the logs/UI
            body = resp.text
            raise HTTPException(status_code=502, detail=f"LLM API error: {resp.status_code} - {body}")

        try:
            data = resp.json()
        except ValueError:
            raise HTTPException(status_code=502, detail=f"LLM API error: invalid JSON body - {resp.text}")

        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            raise HTTPException(status_code=502, detail=f"LLM API error: unexpected response shape - {data}")
        return text, model


async def _call_llm(facts: Dict[str, Any]) -> tuple[str, str]:
    """
    Call the configured LLM using the narrative prompt.
    """
    user_content = _build_prompt(facts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return await _call_llm_messages(messages)


def _build_xai_prompt(ticker: str, technical: Dict[str, Any], fundamental: Dict[str, Any], graph: Dict[str, Any]) -> str:
    facts = {
        "ticker": ticker,
        "technical": technical,
        "fundamental": fundamental,
        "graph": graph,
    }
    return (
        "Generate human-readable explanations for each section using only the facts. "
        "Do not remove any item from lists. Follow the exact output format.\n"
        f"FACTS:\n{json.dumps(facts, indent=2, sort_keys=True, ensure_ascii=True)}"
    )


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _parse_xai_sections(text: str) -> Dict[str, str]:
    cleaned = _strip_code_fences(text)
    match = re.search(r"TECHNICAL:\s*(.*?)\s*FUNDAMENTAL:\s*(.*?)\s*GRAPH:\s*(.*)$", cleaned, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Missing TECHNICAL/FUNDAMENTAL/GRAPH sections")

    technical, fundamental, graph = match.groups()
    return {
        "technical": technical.strip(),
        "fundamental": fundamental.strip(),
        "graph": graph.strip(),
    }


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/generate", response_model=NarrativeResponse)
async def generate_narrative(req: NarrativeRequest):
    """Generate an LLM narrative for a single ticker (on-demand only)."""
    ticker = req.ticker.strip().upper()
    explain_version = req.explanation_facts.get("explain_version", "v1")

    # Check cache first
    cached = _get_cached_narrative(ticker, explain_version)
    if cached:
        return NarrativeResponse(
            narrative=cached["narrative"],
            model_used=cached["model_used"],
            generated_at=cached["generated_at"],
            cached=True,
        )

    # Call LLM
    narrative, model_used = await _call_llm(req.explanation_facts)

    # Cache it
    _set_cached_narrative(ticker, explain_version, narrative, model_used)

    return NarrativeResponse(
        narrative=narrative,
        model_used=model_used,
        generated_at=datetime.now().isoformat(),
        cached=False,
    )


@router.post("/xai", response_model=XaiNarrativeResponse)
async def generate_xai_narratives(req: XaiNarrativeRequest):
    """Generate human-readable XAI narratives on-demand."""
    ticker = req.ticker.strip().upper()
    explain_version = req.explain_version or "v1"

    cached = _get_cached_xai_narrative(ticker, explain_version)
    if cached:
        return XaiNarrativeResponse(
            technical=cached.get("technical", ""),
            fundamental=cached.get("fundamental", ""),
            graph=cached.get("graph", ""),
            model_used=cached.get("model_used", ""),
            generated_at=cached.get("generated_at", ""),
            cached=True,
        )

    user_content = _build_xai_prompt(ticker, req.technical, req.fundamental, req.graph)
    messages = [
        {"role": "system", "content": XAI_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    raw_text, model_used = await _call_llm_messages(messages)
    try:
        parsed = _parse_xai_sections(raw_text)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"LLM API error: invalid response format: {exc}")

    _set_cached_xai_narrative(
        ticker,
        explain_version,
        parsed["technical"],
        parsed["fundamental"],
        parsed["graph"],
        model_used,
    )

    return XaiNarrativeResponse(
        technical=parsed["technical"],
        fundamental=parsed["fundamental"],
        graph=parsed["graph"],
        model_used=model_used,
        generated_at=datetime.now().isoformat(),
        cached=False,
    )


@router.get("/narrative/{ticker}", response_model=NarrativeResponse)
async def get_narrative(ticker: str, explain_version: str = "v1"):
    """Return cached narrative for a ticker (no LLM call)."""
    ticker = ticker.strip().upper()
    cached = _get_cached_narrative(ticker, explain_version)
    if not cached:
        raise HTTPException(status_code=404, detail="No narrative cached for this ticker.")
    return NarrativeResponse(
        narrative=cached["narrative"],
        model_used=cached["model_used"],
        generated_at=cached["generated_at"],
        cached=True,
    )
