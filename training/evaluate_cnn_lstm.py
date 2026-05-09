# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  training/evaluate_cnn_lstm.py                                           ║
# ║  BVMT — CNN-LSTM Evaluation on 2025 Test Set                            ║
# ║                                                                          ║
# ║  PURPOSE: Compute the same metrics as evaluate_tft.py for CNN-LSTM      ║
# ║  so both models can be fairly compared in thesis Chapter 5.             ║
# ║                                                                          ║
# ║  HOW TO RUN:                                                             ║
# ║  python training/evaluate_cnn_lstm.py                                   ║
# ║                                                                          ║
# ║  REQUIRES:                                                               ║
# ║  - models/cnn_lstm_v2_*_best.pt  (saved by train_cnn_lstm.py)          ║
# ║  - results/cnn_lstm_v2_per_stock_stats.json  (normalization stats)      ║
# ║  - data/features/tft_features.csv                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

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
# ══════════════════════════════════════════════════════════════════════════

DATA_PATH   = Path("data/features/tft_features.csv")
MODEL_DIR   = Path("models")
RESULTS_DIR = Path("results")
STATS_PATH  = RESULTS_DIR / "cnn_lstm_v2_per_stock_stats.json"

VAL_END     = "2025-01-01"   # test period starts here
WINDOW_SIZE = 60             # must match training

FEATURE_COLS = [
    "close_price", "open_price", "high_price", "low_price",
    "volume", "daily_return", "ma_5", "ma_20", "ma_50",
    "rsi_14", "volatility_20", "volume_ma_20",
]

print("=" * 60)
print("CNN-LSTM EVALUATION — 2025 Test Set")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — FIND THE BEST MODEL FILE
#
# train_cnn_lstm.py saves with a timestamped filename.
# We find the most recent one automatically so you don't need to
# hardcode the exact filename.
# ══════════════════════════════════════════════════════════════════════════

model_files = sorted(MODEL_DIR.glob("cnn_lstm_v2.pt"))
if not model_files:
    raise FileNotFoundError(
        "No CNN-LSTM model found in models/\n"
        "Run: python training/train_cnn_lstm.py first"
    )

MODEL_PATH = model_files[-1]  # most recent
print(f"Model: {MODEL_PATH}")
print(f"Stats: {STATS_PATH}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD MODEL CHECKPOINT AND ARCHITECTURE CONFIG
#
# The checkpoint saved by train_cnn_lstm.py contains:
#   - model_state: the trained weights
#   - model_config: the exact architecture parameters used
#   - per_stock_stats: normalization means and stds per stock
#   - val_loss, val_acc: from training
#
# We use model_config to recreate the exact same architecture.
# Using a different architecture than what was trained will give
# wrong predictions because the weight shapes won't match.
# ══════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ckpt = torch.load(MODEL_PATH, map_location=device)
cfg  = ckpt["model_config"]

print(f"\nCheckpoint info:")
print(f"  Trained epoch : {ckpt['epoch']}")
print(f"  Val loss      : {ckpt['val_loss']:.4f}")
print(f"  Val accuracy  : {ckpt['val_acc']:.1f}%")
if "acc_up" in ckpt:
    print(f"  Val UP acc    : {ckpt['acc_up']:.1f}%")
else:
    print("  Val UP acc    : N/A (not stored in checkpoint)")

if "acc_down" in ckpt:
    print(f"  Val DOWN acc  : {ckpt['acc_down']:.1f}%")
else:
    print("  Val DOWN acc  : N/A (not stored in checkpoint)")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL DEFINITION
#
# This is the exact same CNNLSTMModel class from train_cnn_lstm.py.
# It must be identical — any change to the architecture means the
# saved weights won't load correctly.
# ══════════════════════════════════════════════════════════════════════════

class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        num_features=12,
        window_size=60,
        num_filters=64,
        num_filters2=128,
        kernel1=5,
        kernel2=3,
        num_filters3=None,
        kernel3=10,
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.use_conv3 = num_filters3 is not None
        self.conv1 = nn.Conv1d(num_features, num_filters, kernel1)
        self.bn1   = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters2, kernel2)
        self.bn2   = nn.BatchNorm1d(num_filters2)
        self.pool  = nn.MaxPool1d(2)
        self.use_proj = self.use_conv3
        if self.use_conv3:
            self.conv3 = nn.Conv1d(num_filters2, num_filters3, kernel3)
            self.bn3   = nn.BatchNorm1d(num_filters3)
            lstm_in = num_filters3
        else:
            lstm_in = num_filters2

        if self.use_proj:
            self.proj  = nn.Linear(lstm_in, lstm_hidden)
            lstm_input_size = lstm_hidden
        else:
            lstm_input_size = lstm_in

        self.lstm  = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(lstm_hidden, 2)
        self.relu    = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.use_conv3:
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.transpose(1, 2)
        if self.use_proj:
            x_proj = self.proj(x)
            lstm_out, _ = self.lstm(x_proj)
            x = lstm_out + x_proj
        else:
            lstm_out, _ = self.lstm(x)
            x = lstm_out
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)


model = CNNLSTMModel(
    num_features = cfg.get("num_features", 12),
    window_size  = cfg.get("window_size", WINDOW_SIZE),
    num_filters  = cfg.get("num_filters", 64),
    num_filters2 = cfg.get("num_filters2", 128),
    num_filters3 = cfg.get("num_filters3", None),
    kernel1      = cfg.get("kernel1", cfg.get("kernel_size", 5)),
    kernel2      = cfg.get("kernel2", 3),
    kernel3      = cfg.get("kernel3", 10),
    lstm_hidden  = cfg.get("lstm_hidden", 64),
    lstm_layers  = cfg.get("lstm_layers", 2),
    dropout      = cfg.get("dropout", 0.3),
).to(device)

model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"\nModel loaded: {sum(p.numel() for p in model.parameters()):,} parameters")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — DATASET FOR TEST SPLIT
#
# We use the per_stock_stats saved during training for normalization.
# Using training statistics (not test statistics) is critical to avoid
# data leakage — in production you would never know the test set stats.
# ══════════════════════════════════════════════════════════════════════════

class BVMTWindowDataset(Dataset):
    def __init__(self, df, window_size=60, per_stock_stats=None):
        self.window_size = window_size
        self.samples = []

        for tid, group in df.groupby("ticker_id"):
            group = group.sort_values("time_idx").reset_index(drop=True)
            tid_int = int(tid)

            if per_stock_stats and str(tid_int) in per_stock_stats:
                s    = per_stock_stats[str(tid_int)]
                mean = np.array(s["mean"], dtype=np.float32)
                std  = np.array(s["std"],  dtype=np.float32)
            else:
                # Fallback: compute from this group
                vals = group[FEATURE_COLS].values.astype(np.float32)
                mean = vals.mean(axis=0)
                std  = vals.std(axis=0)
                std[std < 1e-6] = 1.0

            raw    = group[FEATURE_COLS].values.astype(np.float32)
            normed = (raw - mean) / std
            targets = group["direction_7d"].values

            n = len(group)
            for start in range(n - window_size):
                end = start + window_size
                x = normed[start:end]
                y = int(targets[end - 1])
                if y in (0, 1):
                    self.samples.append((
                        torch.tensor(x, dtype=torch.float32),
                        torch.tensor(y, dtype=torch.long),
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# Load per-stock normalization stats from training
if STATS_PATH.exists():
    with open(STATS_PATH) as f:
        per_stock_stats = json.load(f)
    print(f"Loaded per-stock stats for {len(per_stock_stats)} stocks")
elif "per_stock_stats" in ckpt:
    per_stock_stats = ckpt["per_stock_stats"]
    print("Using per-stock stats from checkpoint")
else:
    per_stock_stats = None
    print("WARNING: No per-stock stats found — using test data stats (suboptimal)")

# Load and prepare data
print("\nLoading test data...")
df = pd.read_csv(DATA_PATH)
df["date"]         = pd.to_datetime(df["date"])
df["direction_7d"] = df["direction_7d"].astype(float).astype(int)
df["ticker_id"]    = df["ticker_id"].astype(int)

# Test split: 2025 data only
# Include enough history for the first window (60 days before VAL_END)
test_df = df[df["date"] >= VAL_END].copy()
print(f"Test rows: {len(test_df):,} ({VAL_END} onwards)")

if len(test_df) == 0:
    raise ValueError(
        f"No test data found after {VAL_END}\n"
        "Check that tft_features.csv contains 2025 data"
    )

print("Building test windows...")
test_ds = BVMTWindowDataset(test_df, WINDOW_SIZE, per_stock_stats)
print(f"Test samples: {len(test_ds):,}")

test_loader = DataLoader(
    test_ds, batch_size=256, shuffle=False, num_workers=0,
)
print(f"Test batches: {len(test_loader)}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — INFERENCE WITH TIMING
# ══════════════════════════════════════════════════════════════════════════

print("\nRunning inference...")
all_preds   = []
all_actuals = []
all_probs   = []

start_time = time.perf_counter()

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_actuals.extend(y.numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())  # P(UP)

end_time = time.perf_counter()
inference_seconds = end_time - start_time
n_samples         = len(all_preds)
ms_per_sample     = (inference_seconds / max(n_samples, 1)) * 1000

print(f"Predictions: {n_samples:,} samples")
print(f"Inference: {inference_seconds:.2f}s total, {ms_per_sample:.3f}ms/sample")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — COMPUTE METRICS (identical to evaluate_tft.py)
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

n_up_actual    = sum(1 for a in all_actuals if a == 1)
n_down_actual  = sum(1 for a in all_actuals if a == 0)
n_up_correct   = sum(1 for a, p in zip(all_actuals, all_preds) if a==1 and p==1)
n_down_correct = sum(1 for a, p in zip(all_actuals, all_preds) if a==0 and p==0)
acc_up   = n_up_correct   / max(n_up_actual, 1) * 100
acc_down = n_down_correct / max(n_down_actual, 1) * 100

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
# SECTION 8 — SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════

results = {
    "model":           "CNN-LSTM-v2",
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
    "val_loss_training": round(float(ckpt.get("val_loss", float("nan"))), 4),
    "val_acc_training":  round(float(ckpt.get("val_acc", float("nan"))), 2),
}

out_path = RESULTS_DIR / "eval_cnn_lstm.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {out_path}")
print()
print("NEXT STEP: python training/compare_models.py")
print("=" * 60)
