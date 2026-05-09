# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  training/evaluate_tft.py                                                ║
# ║  BVMT — TFT Evaluation on 2025 Test Set                                 ║
# ║                                                                          ║
# ║  PURPOSE: Compute directional accuracy, F1, precision, recall,          ║
# ║  confusion matrix, and inference time for the trained TFT model.        ║
# ║  Results saved to results/eval_tft.json for thesis Chapter 5.           ║
# ║                                                                          ║
# ║  HOW TO RUN:                                                             ║
# ║  python training/evaluate_tft.py                                        ║
# ║                                                                          ║
# ║  REQUIRES:                                                               ║
# ║  - models/tft_bvmt_best.ckpt  (downloaded from Lightning AI)            ║
# ║  - data/features/tft_features.csv                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import json
import time
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder

try:
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import CSVLogger
except ImportError:
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import CSVLogger

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
#
# TEST SPLIT: 2025 data — never seen during training or validation.
# This is the honest out-of-sample performance number for your thesis.
# Using 2025 avoids any information leakage from the training process.
# ══════════════════════════════════════════════════════════════════════════

DATA_PATH  = Path("data/features/tft_features.csv")
MODEL_PATH = Path("models/tft_bvmt_best.ckpt")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Must match training exactly
TRAIN_END  = "2024-01-01"
VAL_END    = "2025-01-01"
# Test = everything from 2025-01-01 onwards

MAX_ENCODER_LENGTH    = 60
MAX_PREDICTION_LENGTH = 7

FEATURE_COLS = [
    "close_price", "open_price", "high_price", "low_price",
    "volume", "daily_return", "ma_5", "ma_20", "ma_50",
    "rsi_14", "volatility_20", "volume_ma_20",
]
NON_NUMERIC_COLS = {
    "ticker", "ticker_id", "company_type_id",
    "date", "direction_7d", "time_idx",
}

print("=" * 60)
print("TFT EVALUATION — 2025 Test Set")
print("=" * 60)
print(f"Model : {MODEL_PATH}")
print(f"Test  : {VAL_END} → end of dataset")
print()


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING (same dtype fixes as train_tft.py)
#
# We must reproduce the EXACT same preprocessing as training.
# Any difference in dtype handling or encoding will give wrong predictions.
# ══════════════════════════════════════════════════════════════════════════

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# Force float32 on all feature columns (same as training)
for col in df.columns:
    if col in NON_NUMERIC_COLS:
        continue
    if str(df[col].dtype) == "object":
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    elif str(df[col].dtype) in ("Int8","Int16","Int32","Int64",
                                 "UInt8","UInt16","UInt32","UInt64"):
        df[col] = df[col].astype("float32")
    else:
        try:
            df[col] = df[col].astype("float32")
        except (ValueError, TypeError):
            pass

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(0.0)
df["time_idx"]    = df["time_idx"].astype("int32")
df["direction_7d"] = (df["direction_7d"]
                      .astype(float).astype(int).astype(str))
df = df[df["direction_7d"].isin(["0", "1"])]
df["ticker_id"]       = df["ticker_id"].astype(str)
df["company_type_id"] = df["company_type_id"].astype(str)

print(f"Loaded: {len(df):,} rows, {df['ticker_id'].nunique()} stocks")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATASET RECONSTRUCTION
#
# WHY WE NEED TRAINING DATA FOR TEST EVALUATION:
#   TFT's encoder reads 60 days of history before making a prediction.
#   To predict January 7, 2025, it needs to read November–December 2024.
#   So the test dataset must include all data from 2024-11-01 onwards
#   (the last 60 trading days of the validation period).
#
#   We reconstruct the training_dataset with the same parameters
#   as training so that from_dataset() uses the same encoders,
#   normalizers, and vocabulary. Then we create test_dataset from it.
# ══════════════════════════════════════════════════════════════════════════

print("\nReconstructing datasets...")

# Full history needed for encoder context
train_df = df[df["date"] < TRAIN_END].reset_index(drop=True)

# Test = 2025 data BUT include trailing training data for encoder context
# We include all of 2024 so the encoder has 60 days of context at Jan 2025
test_context_df = df[df["date"] >= TRAIN_END].reset_index(drop=True)

# Final dtype enforcement
for frame in [train_df, test_context_df]:
    for col in FEATURE_COLS:
        if col in frame.columns and frame[col].dtype == object:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype("float32")
    frame["time_idx"]        = frame["time_idx"].astype("int32")
    frame["direction_7d"]    = frame["direction_7d"].astype(str)
    frame["ticker_id"]       = frame["ticker_id"].astype(str)
    frame["company_type_id"] = frame["company_type_id"].astype(str)

# Build training dataset — needed to get correct encoders
training_dataset = TimeSeriesDataSet(
    train_df,
    time_idx              = "time_idx",
    group_ids             = ["ticker_id"],
    target                = "direction_7d",
    min_encoder_length    = MAX_ENCODER_LENGTH // 2,
    max_encoder_length    = MAX_ENCODER_LENGTH,
    min_prediction_length = 1,
    max_prediction_length = MAX_PREDICTION_LENGTH,
    static_categoricals   = ["ticker_id", "company_type_id"],
    time_varying_known_reals   = ["time_idx"],
    time_varying_unknown_reals = FEATURE_COLS,
    target_normalizer     = NaNLabelEncoder(),
    add_relative_time_idx = True,
    add_target_scales     = False,
    add_encoder_length    = True,
    allow_missing_timesteps = True,
)

# Test dataset uses same encoders as training
test_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,
    test_context_df,
    predict            = True,
    stop_randomization = True,
)

test_dataloader = test_dataset.to_dataloader(
    train=False, batch_size=128, num_workers=0,
)

# Count actual 2025 test samples
n_test = sum(
    1 for _, group in test_context_df.groupby("ticker_id")
    if (group["date"] >= VAL_END).sum() > 0
)
print(f"Training samples (for encoder rebuild): {len(training_dataset):,}")
print(f"Test context rows: {len(test_context_df):,}")
print(f"Test batches: {len(test_dataloader)}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — LOAD MODEL AND RUN INFERENCE
#
# TFT outputs raw logits for each class (DOWN=0, UP=1).
# We apply argmax to get the predicted class.
# We also measure inference time per sample for the thesis comparison.
# ══════════════════════════════════════════════════════════════════════════

print(f"\nLoading model from {MODEL_PATH}...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}\n"
        "Download tft_bvmt_best.ckpt from Lightning AI and place in models/"
    )

model = TemporalFusionTransformer.load_from_checkpoint(str(MODEL_PATH))
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Run prediction with timing
print("Running inference on test set...")
start_time = time.perf_counter()

# Trainer needed for predict() — minimal setup, no logging
trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    enable_progress_bar=True,
)

raw_predictions = trainer.predict(model, test_dataloader)

end_time = time.perf_counter()
inference_seconds = end_time - start_time

# Collect all predictions and actuals
all_preds   = []
all_actuals = []

for batch_output in raw_predictions:
    # batch_output is a tuple: (predictions_tensor, x_dict)
    # predictions_tensor shape: [batch, prediction_length, num_classes]
    if isinstance(batch_output, (list, tuple)):
        pred_tensor = batch_output[0]
    else:
        pred_tensor = batch_output

    # Take the first prediction step (day+1) and argmax over classes
    if pred_tensor.dim() == 3:
        # [batch, pred_length, classes] → take step 0
        logits = pred_tensor[:, 0, :]
    else:
        logits = pred_tensor

    preds = logits.argmax(dim=-1).cpu().numpy()
    all_preds.extend(preds.tolist())

# Get actual labels from the test dataloader
for batch in test_dataloader:
    x, (y, _) = batch
    # y is the target tensor — decode from label encoding
    if hasattr(y, "cpu"):
        actuals = y[:, 0].long().cpu().numpy()
    else:
        actuals = np.array(y)[:, 0].astype(int)
    all_actuals.extend(actuals.tolist())

# Align lengths (predictions and actuals must match)
min_len = min(len(all_preds), len(all_actuals))
all_preds   = all_preds[:min_len]
all_actuals = all_actuals[:min_len]

n_samples = len(all_preds)
ms_per_sample = (inference_seconds / max(n_samples, 1)) * 1000

print(f"Predictions: {n_samples:,} samples")
print(f"Inference time: {inference_seconds:.2f}s total, "
      f"{ms_per_sample:.3f}ms per sample")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — COMPUTE METRICS
#
# METRICS EXPLAINED FOR THESIS:
#
# Accuracy: % of predictions that are correct overall
#   = (correct UP predictions + correct DOWN predictions) / total
#
# F1 Score (macro): harmonic mean of precision and recall, averaged
#   across both classes equally. Better than accuracy for imbalanced data.
#   F1=1.0 is perfect, F1=0.5 is random baseline.
#
# Precision (UP): of all times we predicted UP, how often were we right?
# Recall (UP): of all actual UP days, how many did we correctly identify?
#
# Confusion Matrix:
#   [[TN, FP],    TN = correctly predicted DOWN
#    [FN, TP]]    TP = correctly predicted UP
#                 FP = predicted UP but was DOWN (false alarm)
#                 FN = predicted DOWN but was UP (missed opportunity)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST SET METRICS (2025 — never seen during training)")
print("=" * 60)

accuracy  = accuracy_score(all_actuals, all_preds) * 100
f1_macro  = f1_score(all_actuals, all_preds, average="macro")
f1_up     = f1_score(all_actuals, all_preds, average=None)[1]
f1_down   = f1_score(all_actuals, all_preds, average=None)[0]
prec_up   = precision_score(all_actuals, all_preds, pos_label=1)
prec_down = precision_score(all_actuals, all_preds, pos_label=0)
rec_up    = recall_score(all_actuals, all_preds, pos_label=1)
rec_down  = recall_score(all_actuals, all_preds, pos_label=0)
cm        = confusion_matrix(all_actuals, all_preds)

n_up_actual   = sum(1 for a in all_actuals if a == 1)
n_down_actual = sum(1 for a in all_actuals if a == 0)
n_up_correct  = sum(1 for a, p in zip(all_actuals, all_preds) if a == 1 and p == 1)
n_down_correct= sum(1 for a, p in zip(all_actuals, all_preds) if a == 0 and p == 0)
acc_up        = n_up_correct   / max(n_up_actual, 1) * 100
acc_down      = n_down_correct / max(n_down_actual, 1) * 100

# Interpretation for thesis
if   accuracy >= 60: interp = "STRONG — significant alpha signal"
elif accuracy >= 56: interp = "GOOD — meaningful directional signal"
elif accuracy >= 52: interp = "ACCEPTABLE — slight edge over random"
else:                interp = "WEAK — near random baseline (50%)"

print(f"\n  Overall accuracy : {accuracy:.2f}%  [{interp}]")
print(f"  UP accuracy      : {acc_up:.2f}%  ({n_up_correct}/{n_up_actual} correct)")
print(f"  DOWN accuracy    : {acc_down:.2f}%  ({n_down_correct}/{n_down_actual} correct)")
print(f"  F1 macro         : {f1_macro:.4f}")
print(f"  F1 UP            : {f1_up:.4f}")
print(f"  F1 DOWN          : {f1_down:.4f}")
print(f"  Precision UP     : {prec_up:.4f}")
print(f"  Recall UP        : {rec_up:.4f}")
print(f"  Inference time   : {ms_per_sample:.3f} ms/sample")
print()
print("  Confusion Matrix:")
print(f"    Predicted:  DOWN   UP")
print(f"    Actual DOWN: {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"    Actual UP:   {cm[1][0]:5d}  {cm[1][1]:5d}")
print()
print(classification_report(
    all_actuals, all_preds,
    target_names=["DOWN", "UP"],
    digits=4
))


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════

results = {
    "model":           "TFT",
    "test_period":     f"{VAL_END} to end",
    "n_samples":       n_samples,
    "accuracy":        round(accuracy, 4),
    "acc_up":          round(acc_up, 4),
    "acc_down":        round(acc_down, 4),
    "f1_macro":        round(float(f1_macro), 4),
    "f1_up":           round(float(f1_up), 4),
    "f1_down":         round(float(f1_down), 4),
    "precision_up":    round(float(prec_up), 4),
    "recall_up":       round(float(rec_up), 4),
    "precision_down":  round(float(prec_down), 4),
    "recall_down":     round(float(rec_down), 4),
    "confusion_matrix": cm.tolist(),
    "inference_ms_per_sample": round(ms_per_sample, 4),
    "inference_total_seconds": round(inference_seconds, 2),
    "interpretation":  interp,
}

out_path = RESULTS_DIR / "eval_tft.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {out_path}")
print()
print("NEXT STEP: python training/evaluate_cnn_lstm.py")
print("Then:      python training/compare_models.py")
print("=" * 60)
