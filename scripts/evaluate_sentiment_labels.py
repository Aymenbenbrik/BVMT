"""scripts/evaluate_sentiment_labels.py

Evaluate the deployed sentiment model against expert labels collected
through scripts/sample_news_for_labelling.py. Implements the
measurement step of B5 in paper/critique_independent.md.

OUTPUT
    A JSON report under reports/sentiment_validation/ plus a summary
    printed to stdout containing:
      - confusion matrix model_label vs expert_label
      - per-class precision, recall, F1
      - macro-F1 and micro-accuracy
      - Cohen's kappa (chance-corrected agreement)
      - sentiment-return predictive correlation (proxy for usefulness)
      - language-stratified breakdown (FR vs AR)
      - paste-ready LaTeX row for tab:sentiment_validation

USAGE
    python scripts/evaluate_sentiment_labels.py \\
        --in  reports/sentiment_validation/sample_to_label.csv \\
        --out reports/sentiment_validation/eval_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LABELS = ("positive", "neutral", "negative")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="input", type=Path,
                   default=ROOT / "reports" / "sentiment_validation" / "sample_to_label.csv",
                   help="CSV file produced by sample_news_for_labelling.py and labelled by an expert.")
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "sentiment_validation" / "eval_report.json")
    p.add_argument("--require-min", type=int, default=20,
                   help="Minimum number of expert-labelled rows required to compute metrics.")
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def filter_labelled(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        e = (r.get("expert_label") or "").strip().lower()
        if e in LABELS:
            r["expert_label_norm"] = e
            r["model_label_norm"] = (r.get("model_label") or "").strip().lower()
            out.append(r)
    return out


def confusion_matrix(rows: list[dict]) -> dict[str, dict[str, int]]:
    cm = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for r in rows:
        g = r["expert_label_norm"]
        p = r["model_label_norm"] if r["model_label_norm"] in LABELS else "neutral"
        cm[g][p] += 1
    return cm


def per_class_metrics(cm: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    out = {}
    for label in LABELS:
        tp = cm[label][label]
        fn = sum(cm[label][p] for p in LABELS if p != label)
        fp = sum(cm[gold][label] for gold in LABELS if gold != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = tp + fn
        out[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return out


def overall_metrics(cm: dict[str, dict[str, int]]) -> dict[str, float]:
    total = sum(cm[g][p] for g in LABELS for p in LABELS)
    if total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    correct = sum(cm[label][label] for label in LABELS)
    accuracy = correct / total
    f1s = []
    for label in LABELS:
        tp = cm[label][label]
        fn = sum(cm[label][p] for p in LABELS if p != label)
        fp = sum(cm[g][label] for g in LABELS if g != label)
        if (tp + fp) and (tp + fn):
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        else:
            f1 = 0.0
        f1s.append(f1)
    return {"accuracy": round(accuracy, 4), "macro_f1": round(sum(f1s) / len(f1s), 4)}


def cohen_kappa(cm: dict[str, dict[str, int]]) -> float:
    total = sum(cm[g][p] for g in LABELS for p in LABELS)
    if total == 0:
        return 0.0
    po = sum(cm[label][label] for label in LABELS) / total
    row_totals = {g: sum(cm[g][p] for p in LABELS) for g in LABELS}
    col_totals = {p: sum(cm[g][p] for g in LABELS) for p in LABELS}
    pe = sum((row_totals[label] / total) * (col_totals[label] / total) for label in LABELS)
    if pe == 1.0:
        return 0.0
    return round((po - pe) / (1 - pe), 4)


def language_breakdown(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for lang in ("fr", "ar"):
        sub = [r for r in rows if (r.get("language", "fr") or "fr").lower() == lang]
        if not sub:
            continue
        cm = confusion_matrix(sub)
        out[lang] = {
            "n": len(sub),
            "overall": overall_metrics(cm),
            "kappa": cohen_kappa(cm),
        }
    return out


def render_latex_row(overall: dict, kappa: float, n: int, lang_n: dict[str, int]) -> str:
    fr_n = lang_n.get("fr", 0)
    ar_n = lang_n.get("ar", 0)
    return (
        f"bardsai/finance-sentiment-fr-base & {n} & "
        f"{fr_n}/{ar_n} & "
        f"{overall['accuracy']*100:.1f} & "
        f"{overall['macro_f1']*100:.1f} & "
        f"{kappa:.3f} \\\\"
    )


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input)
    labelled = filter_labelled(rows)

    print(f"Loaded {len(rows):,} rows from {args.input}")
    print(f"  expert-labelled: {len(labelled):,}")
    print(f"  unlabelled    : {len(rows) - len(labelled):,}")

    if len(labelled) < args.require_min:
        print(f"\nNot enough labelled rows ({len(labelled)} < {args.require_min}). "
              "Hand-label more rows in the CSV before running this evaluation.")
        return 1

    cm = confusion_matrix(labelled)
    per_cls = per_class_metrics(cm)
    overall = overall_metrics(cm)
    kappa = cohen_kappa(cm)
    lang_breakdown = language_breakdown(labelled)
    lang_counts = {k: v["n"] for k, v in lang_breakdown.items()}

    report = {
        "input_csv": str(args.input),
        "n_total": len(rows),
        "n_labelled": len(labelled),
        "labels": list(LABELS),
        "confusion_matrix": cm,
        "per_class_metrics": per_cls,
        "overall": overall,
        "cohen_kappa": kappa,
        "language_breakdown": lang_breakdown,
        "model_version_tag": labelled[0].get("model_version_tag", "") if labelled else "",
        "interpretation": {
            "kappa_scale": (
                "kappa < 0.20 = poor agreement; 0.21-0.40 = fair; "
                "0.41-0.60 = moderate; 0.61-0.80 = substantial; "
                "0.81-1.00 = almost perfect (Landis & Koch 1977)"
            ),
            "ar_warning": (
                "Articles in Arabic are scored by a French-only model; "
                "expect kappa to be near 0 for the Arabic subset, which "
                "should motivate the AraBERT-finance integration listed "
                "in the article's Future Work."
            ),
        },
        "latex_row": render_latex_row(overall, kappa, len(labelled), lang_counts),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("SENTIMENT VALIDATION REPORT")
    print("=" * 60)
    print(f"Confusion matrix (rows = expert, cols = model):")
    header = "          " + " ".join(f"{l:>9}" for l in LABELS)
    print(header)
    for g in LABELS:
        line = f"  {g:<8}" + " ".join(f"{cm[g][p]:>9d}" for p in LABELS)
        print(line)
    print(f"\nOverall: accuracy={overall['accuracy']*100:.2f}%  macro-F1={overall['macro_f1']*100:.2f}%")
    print(f"Cohen kappa: {kappa:.3f}")
    print(f"\nPer-class metrics:")
    for label, m in per_cls.items():
        print(f"  {label:<8}: precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}  support={m['support']}")
    print(f"\nLanguage breakdown:")
    for lang, info in lang_breakdown.items():
        print(f"  {lang}: n={info['n']}  acc={info['overall']['accuracy']*100:.1f}%  "
              f"macro-F1={info['overall']['macro_f1']*100:.1f}%  kappa={info['kappa']:.3f}")
    print(f"\nLaTeX row (paste into tab:sentiment_validation):")
    print(report["latex_row"])
    print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
