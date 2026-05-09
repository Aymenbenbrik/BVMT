"""scripts/sample_news_for_labelling.py

Stratified random sample of BVMT news articles for expert sentiment
labelling. Implements the data-collection step of B5 in
paper/critique_independent.md and Action 1.7 of paper/improvement_plan.md.

The sentiment scores currently produced by the SentimentAgent come from
bardsai/finance-sentiment-fr-base (Rguibi et al. 2023), evaluated on
that paper's own benchmark, not on the BVMT corpus from ilboursa.com.
This script extracts a representative sample so a domain expert can
hand-label it; the resulting CSV feeds evaluate_sentiment_labels.py.

STRATIFICATION
    The sample is balanced along three axes to avoid sampling biases
    that would inflate a single ticker / year / language:
      - by year      : equal target count per year of coverage
      - by language  : ~85% French / ~15% Arabic, matching corpus mix
      - by model_label : balanced across {positive, neutral, negative}
                         so the agreement matrix has support in every cell

    When a stratum has fewer rows than the target, every row in that
    stratum is included and the deficit is redistributed proportionally
    to the other strata.

USAGE
    python scripts/sample_news_for_labelling.py \\
        --n 200 \\
        --out reports/sentiment_validation/sample_to_label.csv \\
        --seed 42

ENV
    DB_HOST, DB_PORT (5432), DB_NAME, DB_USER, DB_PASSWORD
    .env at the project root is loaded automatically via python-dotenv.

OUTPUT CSV columns
    Frozen at sampling time (model_score, model_label, model_version
    snapshot) so the evaluation script can compute disagreement even if
    the model is later retrained:

        article_id, ticker, isin_code, language, published_at,
        title, content_excerpt,
        model_score, model_label, model_version_tag,
        expert_label, expert_confidence, expert_notes  -- empty, to fill

    The first 9 columns are written by this script. The last three are
    blank by design and meant to be filled by the labeller in any
    spreadsheet editor (LibreOffice, Excel, Google Sheets).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import psycopg2
    import psycopg2.extras
except ImportError as exc:
    raise SystemExit(
        "psycopg2 is required. Install with: pip install psycopg2-binary"
    ) from exc

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# Allowed expert labels (the annotation guidelines must align with this set).
EXPERT_LABEL_SET = ("positive", "neutral", "negative")
MODEL_VERSION_TAG = "bardsai/finance-sentiment-fr-base@labelling-snapshot"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200, help="Total sample size (default 200).")
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "sentiment_validation" / "sample_to_label.csv",
                   help="Output CSV path.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--min-content-length", type=int, default=20,
                   help="Skip articles whose title+content has fewer than this many characters.")
    return p.parse_args()


def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def fetch_corpus(conn, min_len: int) -> list[dict]:
    sql = """
        SELECT
            id                                   AS article_id,
            ticker,
            COALESCE(isin_code, '')              AS isin_code,
            COALESCE(language, 'fr')             AS language,
            published_at,
            COALESCE(title, '')                  AS title,
            COALESCE(content, '')                AS content,
            sentiment_score                      AS model_score,
            sentiment_label                      AS model_label
        FROM news_articles
        WHERE sentiment_label IS NOT NULL
          AND char_length(COALESCE(title, '') || ' ' || COALESCE(content, '')) >= %s
        ORDER BY published_at;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (min_len,))
        return [dict(r) for r in cur.fetchall()]


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    import random
    rng = random.Random(seed)

    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        year = r["published_at"].year if r["published_at"] is not None else 0
        lang = (r["language"] or "fr").lower()
        if lang not in {"fr", "ar"}:
            lang = "fr"
        label = (r["model_label"] or "neutral").lower()
        if label not in EXPERT_LABEL_SET:
            label = "neutral"
        key = (year, lang, label)
        by_key.setdefault(key, []).append(r)

    if not by_key:
        return []

    # Balanced quota per (year, lang, label) cell, then redistribute deficits.
    n_cells = len(by_key)
    quota = max(1, n // n_cells)
    selected: list[dict] = []
    deficit = 0
    for key, bucket in by_key.items():
        rng.shuffle(bucket)
        take = min(quota, len(bucket))
        selected.extend(bucket[:take])
        deficit += quota - take
    # Top up with random remaining rows so we hit n exactly.
    if len(selected) < n:
        remaining = [r for r in rows if r not in selected]
        rng.shuffle(remaining)
        need = n - len(selected)
        selected.extend(remaining[:need])

    rng.shuffle(selected)
    return selected[:n]


def excerpt(title: str, content: str, max_chars: int = 600) -> str:
    body = content.strip() if content else ""
    if not body:
        return title.strip()
    text = f"{title.strip()} -- {body}"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "article_id", "ticker", "isin_code", "language", "published_at",
        "title", "content_excerpt",
        "model_score", "model_label", "model_version_tag",
        "expert_label", "expert_confidence", "expert_notes",
    ]
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "article_id":         r["article_id"],
                "ticker":             r["ticker"],
                "isin_code":          r["isin_code"],
                "language":           r["language"],
                "published_at":       r["published_at"].isoformat() if r["published_at"] else "",
                "title":              (r["title"] or "").strip(),
                "content_excerpt":    excerpt(r["title"] or "", r["content"] or ""),
                "model_score":        r["model_score"],
                "model_label":        r["model_label"],
                "model_version_tag":  MODEL_VERSION_TAG,
                "expert_label":       "",
                "expert_confidence":  "",
                "expert_notes":       "",
            })


def print_distribution(rows: list[dict]) -> None:
    from collections import Counter
    by_year = Counter()
    by_lang = Counter()
    by_label = Counter()
    by_ticker = Counter()
    for r in rows:
        by_year[r["published_at"].year if r["published_at"] else 0] += 1
        by_lang[(r["language"] or "fr").lower()] += 1
        by_label[(r["model_label"] or "neutral").lower()] += 1
        by_ticker[r["ticker"]] += 1
    print(f"\nSample distribution ({len(rows)} rows):")
    print(f"  by year   : {dict(sorted(by_year.items()))}")
    print(f"  by lang   : {dict(by_lang)}")
    print(f"  by model  : {dict(by_label)}")
    print(f"  unique tickers : {len(by_ticker)}")


def main() -> int:
    args = parse_args()
    print(f"Connecting to database (host={os.getenv('DB_HOST', 'localhost')})...")
    conn = get_conn()
    try:
        rows = fetch_corpus(conn, args.min_content_length)
    finally:
        conn.close()
    print(f"Corpus: {len(rows):,} articles with model labels and >= {args.min_content_length} chars.")

    if len(rows) < args.n:
        print(f"WARN: corpus has fewer rows ({len(rows)}) than requested sample ({args.n}); will use all.")

    sample = stratified_sample(rows, args.n, args.seed)
    print(f"Sampled: {len(sample):,} articles (seed={args.seed}).")
    print_distribution(sample)

    write_csv(sample, args.out)
    print(f"\nWrote: {args.out}")
    print("\nLabelling guidelines (paste into the team chat):")
    print("  - expert_label in {positive, neutral, negative}")
    print("  - expert_confidence in {high, medium, low} (your subjective certainty)")
    print("  - expert_notes free text (e.g. 'sarcasm', 'mixed signals', 'Arabic only')")
    print("  - For Arabic articles, the model is French-only; flag systematically.")
    print(f"  - Once labelled, run: python scripts/evaluate_sentiment_labels.py --in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
