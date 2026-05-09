"""training/cnn_lstm_seed_sweep.py

CNN-LSTM seed-sweep harness for the "ceiling" experiment (B3 of
paper/critique_independent.md). Spawns N independent training runs of
training/train_cnn_lstm.py with different seeds, then aggregates the
per-seed test accuracies into mean +/- 95% bootstrap CI, median, top-
quartile, and a Wilcoxon test against the always-DOWN majority-class
baseline (58.1%).

This script does NOT replace the existing 4-run table in the article.
Its purpose is to produce the statistically powered evidence the
article's "ceiling" claim requires (N>=20 seeds), as identified in
the independent critique. The article's "ceiling" wording must be
softened until this sweep has run; see paper/improvement_plan.md
Action 1.4b for the protocol.

USAGE
    # full training, 20 seeds, default range [0, 19]
    python training/cnn_lstm_seed_sweep.py --n 20 --smoke false

    # smoke test, 3 seeds (sanity check the harness before a full run)
    python training/cnn_lstm_seed_sweep.py --n 3 --smoke true

    # custom seed list
    python training/cnn_lstm_seed_sweep.py --seeds 7,42,123,2024 --smoke false

OUTPUT
    Each individual seed produces results/results_cnnlstm_v2_*_seed<S>.json
    (written by train_cnn_lstm.py). At the end, this script writes a
    summary file at results/cnn_lstm_sweep_<timestamp>.json with the
    aggregate statistics and prints a paste-ready LaTeX row.

NOTES
    - On a CPU-only machine each seed takes ~1-2h on the full dataset;
      run smoke=true first to confirm the harness works.
    - The per-seed JSON file is glob-matched by the run_name suffix
      _seed<S>; do not delete previous seed runs unless you want to
      re-run them.
    - The Wilcoxon test uses the per-seed test_acc values against the
      constant always-DOWN baseline 58.1% (one-sample, two-sided).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
TRAIN_SCRIPT = ROOT / "training" / "train_cnn_lstm.py"

# Conservative bound from Section V.A of the article. If the top-quartile
# of seed test_acc remains below this value, the article's CNN-LSTM
# "ceiling" claim is empirically defensible.
CEILING_CANDIDATE_PCT = 56.0
ALWAYS_DOWN_BASELINE_PCT = 58.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--n", type=int, default=20, help="Number of seeds (default 20). Seeds are 0..n-1.")
    g.add_argument("--seeds", type=str, default=None, help="Explicit comma-separated seed list, overrides --n.")
    p.add_argument("--smoke", choices=["true", "false"], default="false",
                   help="Pass through to BVMT_SMOKE. 'true' = quick sanity, 'false' = full training.")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter used for child runs (default: current).")
    p.add_argument("--bootstrap", type=int, default=10_000,
                   help="Bootstrap samples for the 95%% CI of the mean (default 10,000).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned commands without executing them.")
    return p.parse_args()


def resolve_seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(s) for s in args.seeds.split(",") if s.strip()]
    return list(range(args.n))


def find_seed_result(seed: int) -> Path | None:
    """Return the most recent results JSON written for this seed, if any."""
    pattern = f"results_cnnlstm_v2_*_seed{seed}.json"
    candidates = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_one_seed(seed: int, smoke: str, python_bin: str, dry_run: bool) -> tuple[int, dict | None]:
    env = os.environ.copy()
    env["BVMT_SEED"] = str(seed)
    env["BVMT_SMOKE"] = smoke
    cmd = [python_bin, str(TRAIN_SCRIPT)]
    print(f"\n--- seed {seed} : launching ---", flush=True)
    print("  cmd  :", " ".join(cmd))
    print("  env  : BVMT_SEED=", seed, "BVMT_SMOKE=", smoke, sep="")
    if dry_run:
        return seed, None
    t0 = time.time()
    try:
        subprocess.run(cmd, env=env, check=True, cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        print(f"  seed {seed} FAILED with rc={exc.returncode}; continuing.", flush=True)
        return seed, None
    elapsed = time.time() - t0
    print(f"  seed {seed} done in {elapsed/60:.1f} min", flush=True)
    f = find_seed_result(seed)
    if f is None:
        print(f"  WARN: no results JSON found for seed {seed}", flush=True)
        return seed, None
    with open(f) as h:
        return seed, json.load(h)


def bootstrap_ci(values: list[float], n_resamples: int, ci: float = 0.95, rng_seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(values, dtype=float)
    boots = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return lo, hi


def wilcoxon_one_sample(values: list[float], reference: float) -> dict:
    """Wilcoxon signed-rank test of (values - reference) against zero.

    Lazily imports scipy so the harness itself does not require it for
    the launch step. If scipy is unavailable, return a stub indicating
    the test could not be run.
    """
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except ImportError:
        return {"available": False, "reason": "scipy not installed"}
    diffs = [v - reference for v in values]
    if all(abs(d) < 1e-12 for d in diffs):
        return {"available": True, "p_value": 1.0, "statistic": 0.0, "note": "all diffs are zero"}
    res = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
    return {
        "available": True,
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "alternative": "two-sided",
    }


def summarise(per_seed: list[tuple[int, dict | None]], bootstrap_n: int) -> dict:
    test_accs = [r["test_acc"] for s, r in per_seed if r is not None]
    val_accs = [r.get("best_val_acc") for s, r in per_seed if r is not None]
    seeds_ok = [s for s, r in per_seed if r is not None]
    seeds_fail = [s for s, r in per_seed if r is None]

    if not test_accs:
        return {"error": "no successful runs", "fail_seeds": seeds_fail}

    mean = statistics.fmean(test_accs)
    median = statistics.median(test_accs)
    stdev = statistics.stdev(test_accs) if len(test_accs) > 1 else 0.0
    q1 = float(np.quantile(test_accs, 0.25))
    q3 = float(np.quantile(test_accs, 0.75))
    lo, hi = (
        bootstrap_ci(test_accs, n_resamples=bootstrap_n) if len(test_accs) > 1 else (mean, mean)
    )

    wilc_random = wilcoxon_one_sample(test_accs, 50.0)
    wilc_alwaysdown = wilcoxon_one_sample(test_accs, ALWAYS_DOWN_BASELINE_PCT)

    ceiling_holds = q3 < CEILING_CANDIDATE_PCT
    return {
        "n_seeds_attempted": len(per_seed),
        "n_seeds_ok": len(seeds_ok),
        "n_seeds_fail": len(seeds_fail),
        "fail_seeds": seeds_fail,
        "test_acc": {
            "values": test_accs,
            "seeds": seeds_ok,
            "mean": round(mean, 3),
            "median": round(median, 3),
            "stdev": round(stdev, 3),
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "ci95_lo": round(lo, 3),
            "ci95_hi": round(hi, 3),
        },
        "val_acc_means": [round(v, 3) for v in val_accs if v is not None],
        "tests": {
            "wilcoxon_vs_random_50": wilc_random,
            "wilcoxon_vs_always_down_58_1": wilc_alwaysdown,
        },
        "ceiling": {
            "candidate_pct": CEILING_CANDIDATE_PCT,
            "q3_pct": round(q3, 3),
            "holds_q3_below_candidate": ceiling_holds,
            "interpretation": (
                "If holds_q3_below_candidate is True, the article's "
                "'CNN-LSTM ceiling' claim is empirically defensible at the "
                "candidate level. If False, the ceiling claim must be "
                "softened or the candidate raised."
            ),
        },
    }


def render_latex_row(summary: dict) -> str:
    """Paste-ready LaTeX row for a future tab:cnn_lstm_seed_sweep table."""
    if "test_acc" not in summary:
        return "% no successful runs; sweep failed"
    a = summary["test_acc"]
    p = summary["tests"]["wilcoxon_vs_always_down_58_1"]
    p_str = f"{p['p_value']:.3g}" if p.get("available") and "p_value" in p else "n/a"
    return (
        f"CNN-LSTM v2 (N={summary['n_seeds_ok']}) & "
        f"{a['mean']:.2f} & "
        f"[{a['ci95_lo']:.2f}, {a['ci95_hi']:.2f}] & "
        f"{a['median']:.2f} & "
        f"{a['q3']:.2f} & "
        f"{p_str} \\\\"
    )


def main() -> int:
    args = parse_args()
    seeds = resolve_seed_list(args)
    print(f"Sweep plan: {len(seeds)} seed(s) -> {seeds}")
    print(f"  smoke={args.smoke}  python={args.python}")
    if args.dry_run:
        print("Dry run mode -- not executing children.")
    RESULTS_DIR.mkdir(exist_ok=True)
    results = []
    for s in seeds:
        results.append(run_one_seed(s, args.smoke, args.python, args.dry_run))

    if args.dry_run:
        print("\nDry run complete; no summary written.")
        return 0

    summary = summarise(results, bootstrap_n=args.bootstrap)
    summary["meta"] = {
        "timestamp": datetime.now().isoformat(),
        "seeds": seeds,
        "smoke": args.smoke,
        "bootstrap_samples": args.bootstrap,
        "ceiling_candidate_pct": CEILING_CANDIDATE_PCT,
        "always_down_baseline_pct": ALWAYS_DOWN_BASELINE_PCT,
    }
    summary["latex_row"] = render_latex_row(summary)

    out = RESULTS_DIR / f"cnn_lstm_sweep_{datetime.now():%Y%m%d_%H%M}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, indent=2))
    print("\nLaTeX row (paste into tab:cnn_lstm_seed_sweep):")
    print(summary["latex_row"])
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
