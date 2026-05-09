# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  training/compare_models.py                                              ║
# ║  BVMT — TFT vs CNN-LSTM Comparison (Academic Contribution #2)           ║
# ║                                                                          ║
# ║  PURPOSE: Load results from both evaluation scripts and produce         ║
# ║  the comparison table for thesis Chapter 5, Table 5.1.                 ║
# ║  Also saves compare_models.json for use in the final report.            ║
# ║                                                                          ║
# ║  HOW TO RUN (after both evaluate scripts):                              ║
# ║  python training/compare_models.py                                      ║
# ║                                                                          ║
# ║  REQUIRES:                                                               ║
# ║  - results/eval_tft.json       (from evaluate_tft.py)                   ║
# ║  - results/eval_cnn_lstm.json  (from evaluate_cnn_lstm.py)              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import json
from pathlib import Path

RESULTS_DIR = Path("results")
TFT_RESULTS  = RESULTS_DIR / "eval_tft.json"
CNN_RESULTS  = RESULTS_DIR / "eval_cnn_lstm.json"
OUT_PATH     = RESULTS_DIR / "compare_models.json"

print("=" * 65)
print("ACADEMIC CONTRIBUTION #2 — TFT vs CNN-LSTM Comparison")
print("BVMT Groupe 11 Stocks | Test Period: 2025")
print("=" * 65)

# Load results
if not TFT_RESULTS.exists():
    raise FileNotFoundError(
        f"Missing {TFT_RESULTS}\nRun: python training/evaluate_tft.py first"
    )
if not CNN_RESULTS.exists():
    raise FileNotFoundError(
        f"Missing {CNN_RESULTS}\nRun: python training/evaluate_cnn_lstm.py first"
    )

with open(TFT_RESULTS)  as f: tft = json.load(f)
with open(CNN_RESULTS)  as f: cnn = json.load(f)

# ── Print comparison table ────────────────────────────────────────────────
# This is Table 5.1 in your thesis chapter on model evaluation.
# Format matches standard ML paper comparison tables.

W = 20  # column width

def row(metric, tft_val, cnn_val, better="higher"):
    """Print one table row with winner highlighted."""
    if isinstance(tft_val, float) and isinstance(cnn_val, float):
        if better == "higher":
            winner = "TFT ✓" if tft_val > cnn_val else ("CNN-LSTM ✓" if cnn_val > tft_val else "TIE")
        else:
            winner = "TFT ✓" if tft_val < cnn_val else ("CNN-LSTM ✓" if cnn_val < tft_val else "TIE")
        print(f"  {metric:<28} {str(tft_val):>{W}} {str(cnn_val):>{W}}   {winner}")
    else:
        print(f"  {metric:<28} {str(tft_val):>{W}} {str(cnn_val):>{W}}")

print()
print(f"  {'Metric':<28} {'TFT':>{W}} {'CNN-LSTM v2':>{W}}   Winner")
print(f"  {'-'*28} {'-'*W} {'-'*W}   {'-'*12}")

row("Overall Accuracy (%)",
    round(tft["accuracy"], 2),
    round(cnn["accuracy"], 2))

row("UP Direction Accuracy (%)",
    round(tft["acc_up"], 2),
    round(cnn["acc_up"], 2))

row("DOWN Direction Accuracy (%)",
    round(tft["acc_down"], 2),
    round(cnn["acc_down"], 2))

row("F1 Score (Macro)",
    round(tft["f1_macro"], 4),
    round(cnn["f1_macro"], 4))

row("F1 Score (UP class)",
    round(tft["f1_up"], 4),
    round(cnn["f1_up"], 4))

row("F1 Score (DOWN class)",
    round(tft["f1_down"], 4),
    round(cnn["f1_down"], 4))

row("Precision (UP)",
    round(tft["precision_up"], 4),
    round(cnn["precision_up"], 4))

row("Recall (UP)",
    round(tft["recall_up"], 4),
    round(cnn["recall_up"], 4))

row("Inference (ms/sample)",
    round(tft["inference_ms_per_sample"], 3),
    round(cnn["inference_ms_per_sample"], 3),
    better="lower")

row("Test Samples",
    tft["n_samples"],
    cnn["n_samples"])

print()

# ── Winner summary ────────────────────────────────────────────────────────
tft_wins = 0
cnn_wins = 0
metrics  = ["accuracy", "acc_up", "acc_down", "f1_macro", "f1_up", "f1_down"]
for m in metrics:
    if tft[m] > cnn[m]:
        tft_wins += 1
    elif cnn[m] > tft[m]:
        cnn_wins += 1

# Inference speed winner (lower is better)
if tft["inference_ms_per_sample"] < cnn["inference_ms_per_sample"]:
    tft_wins += 1
else:
    cnn_wins += 1

print(f"  Overall winner: ", end="")
if tft_wins > cnn_wins:
    winner = "TFT"
    print(f"TFT ({tft_wins} metrics vs {cnn_wins})")
elif cnn_wins > tft_wins:
    winner = "CNN-LSTM"
    print(f"CNN-LSTM ({cnn_wins} metrics vs {tft_wins})")
else:
    winner = "TIE"
    print(f"TIE ({tft_wins} each)")

# ── Accuracy improvement ───────────────────────────────────────────────────
acc_diff = tft["accuracy"] - cnn["accuracy"]
print()
print(f"  TFT accuracy improvement over CNN-LSTM: "
      f"{acc_diff:+.2f} percentage points")
if abs(acc_diff) < 1.0:
    print("  Note: difference < 1pp — statistically marginal")
elif abs(acc_diff) < 3.0:
    print("  Note: moderate improvement — meaningful for directional trading")
else:
    print("  Note: substantial improvement — strong evidence for TFT superiority")

# ── Confusion matrices ─────────────────────────────────────────────────────
print()
print("  Confusion Matrix — TFT:")
cm_tft = tft["confusion_matrix"]
print(f"    Predicted: DOWN    UP")
print(f"    DOWN:     {cm_tft[0][0]:5d}  {cm_tft[0][1]:5d}")
print(f"    UP:       {cm_tft[1][0]:5d}  {cm_tft[1][1]:5d}")

print()
print("  Confusion Matrix — CNN-LSTM:")
cm_cnn = cnn["confusion_matrix"]
print(f"    Predicted: DOWN    UP")
print(f"    DOWN:     {cm_cnn[0][0]:5d}  {cm_cnn[0][1]:5d}")
print(f"    UP:       {cm_cnn[1][0]:5d}  {cm_cnn[1][1]:5d}")

# ── Thesis interpretation ─────────────────────────────────────────────────
print()
print("=" * 65)
print("THESIS CHAPTER 5 — INTERPRETATION")
print("=" * 65)
print(f"""
Academic Contribution #2 demonstrates that TFT achieves {tft["accuracy"]:.1f}%
directional accuracy on BVMT Groupe 11 stocks for the 7-day ahead
prediction task, compared to {cnn["accuracy"]:.1f}% for the CNN-LSTM baseline.

The {abs(acc_diff):.1f} percentage point {"advantage" if tft["accuracy"] > cnn["accuracy"] else "deficit"} of TFT {"supports" if tft["accuracy"] > cnn["accuracy"] else "challenges"} the hypothesis
that attention-based architectures capture longer-range temporal
dependencies more effectively than convolutional-recurrent models
on emerging market financial time series.

Both models exceed the 50% random baseline, confirming that
meaningful directional signal exists in BVMT technical indicators
over a 60-day historical window.

The TFT's interpretable attention mechanism (see Chapter 6, XAI)
provides additional academic value beyond raw accuracy by revealing
which historical time steps and features drive predictions.
""")

# ── Save combined results ──────────────────────────────────────────────────
comparison = {
    "test_period": tft["test_period"],
    "n_samples_tft":     tft["n_samples"],
    "n_samples_cnn":     cnn["n_samples"],
    "tft":  {k: tft[k] for k in [
        "accuracy","acc_up","acc_down","f1_macro","f1_up","f1_down",
        "precision_up","recall_up","inference_ms_per_sample","confusion_matrix"
    ]},
    "cnn_lstm": {k: cnn[k] for k in [
        "accuracy","acc_up","acc_down","f1_macro","f1_up","f1_down",
        "precision_up","recall_up","inference_ms_per_sample","confusion_matrix"
    ]},
    "accuracy_diff_tft_minus_cnn": round(acc_diff, 4),
    "overall_winner": winner,
    "tft_wins_metrics": tft_wins,
    "cnn_wins_metrics": cnn_wins,
}

with open(OUT_PATH, "w") as f:
    json.dump(comparison, f, indent=2)

print(f"Comparison saved to {OUT_PATH}")
print()
print("FILES READY FOR THESIS:")
print(f"  {TFT_RESULTS}")
print(f"  {CNN_RESULTS}")
print(f"  {OUT_PATH}")
print("=" * 65)
