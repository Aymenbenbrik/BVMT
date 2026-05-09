# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  training/train_cnn_lstm.py  — VERSION 2                                ║
# ║  BVMT — CNN-LSTM Training Script (Enhanced)                             ║
# ║                                                                          ║
# ║  UPDATE SECTION (V2 vs V1):                                              ║
# ║  FIX 1 — Per-stock normalization (was global, causing scale chaos)      ║
# ║  FIX 2 — Learning rate 0.001 → 0.0003 + linear warmup                  ║
# ║  FIX 3 — Class weights were inverted (backwards formula)                ║
# ║  FIX 4 — Window construction now strictly per-stock                     ║
# ║  FIX 5 — Added 3rd CNN layer with kernel=10 (2-week patterns)          ║
# ║  FIX 6 — Added residual connection around LSTM (gradient vanishing)     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import inspect
import json
import os
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

print("PyTorch version:", torch.__version__)
print("CUDA available :", torch.cuda.is_available())


# ══════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY — seed injection for the seed-sweep harness (B3)
#
# When BVMT_SEED is set in the environment, the script becomes reproducible
# at the level pytorch can guarantee on a given hardware (cuDNN may still
# add small stochasticity across drivers). The seed is also embedded in
# run_name so the seed-sweep harness can collect per-seed result files.
#
# When BVMT_SEED is absent, the previous non-deterministic behavior is
# preserved.
#
# When BVMT_SMOKE is set ("0" or "false"), it overrides SMOKE_TEST below.
# This lets the sweep harness force a full run without editing the file.
# ══════════════════════════════════════════════════════════════════════════

BVMT_SEED_ENV = os.environ.get("BVMT_SEED")
SEED = int(BVMT_SEED_ENV) if BVMT_SEED_ENV is not None else None
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # Best-effort determinism on cuDNN; not a guarantee across hardware.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"BVMT_SEED={SEED} -> deterministic mode (numpy, torch, cuDNN)")
else:
    print("BVMT_SEED not set -> non-deterministic mode (legacy)")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# BVMT_SMOKE env var overrides this default. Accepted "true" values:
# "0" / "false" / "no" (case-insensitive) -> full training; anything else
# (including unset) -> the file default below.
_SMOKE_DEFAULT = True
_smoke_env = os.environ.get("BVMT_SMOKE")
if _smoke_env is not None and _smoke_env.strip().lower() in {"0", "false", "no"}:
    SMOKE_TEST = False
else:
    SMOKE_TEST = _SMOKE_DEFAULT

DATA_PATH = Path("data/features/tft_features.csv")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 60
PRED_HORIZON = 7

# CNN — 3 layers now
NUM_FILTERS = 64
NUM_FILTERS2 = 128
NUM_FILTERS3 = 128
KERNEL_SIZE1 = 5  # 1 trading week
KERNEL_SIZE2 = 3  # pattern combination
KERNEL_SIZE3 = 10  # 2 trading weeks — FIX 5

# LSTM
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
DROPOUT = 0.3

# FIX 2: Lower LR — was 0.001, now 0.0003
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2

if SMOKE_TEST:
    SMOKE_STOCKS = 3
    MAX_EPOCHS = 3
    BATCH_SIZE = 32
    print("=" * 55)
    print("SMOKE TEST MODE — 3 stocks, 3 epochs")
    print("=" * 55)
else:
    MAX_EPOCHS = 40
    BATCH_SIZE = 128
    print("=" * 55)
    print("FULL TRAINING MODE — all stocks, 40 epochs")
    print("=" * 55)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
    if vram_gb < 6 and not SMOKE_TEST:
        BATCH_SIZE = min(BATCH_SIZE, 64)
        NUM_FILTERS = 32
        NUM_FILTERS2 = 64
        NUM_FILTERS3 = 64
        LSTM_HIDDEN = 32
        print("Low VRAM: model size reduced")
else:
    DEVICE = torch.device("cpu")
    print("No GPU — CPU (smoke test only)")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET WITH PER-STOCK NORMALIZATION
#
# FIX 1 + FIX 4: Normalize each stock by its OWN mean and std.
# ══════════════════════════════════════════════════════════════════════════


class BVMTWindowDataset(Dataset):
    FEATURE_COLS = [
        "close_price",
        "open_price",
        "high_price",
        "low_price",
        "volume",
        "daily_return",
        "ma_5",
        "ma_20",
        "ma_50",
        "rsi_14",
        "volatility_20",
        "volume_ma_20",
    ]

    def __init__(self, df, window_size=60, fit_scaler=True, per_stock_stats=None):
        self.window_size = window_size
        self.samples = []

        if fit_scaler:
            # Compute per-stock stats from training data only
            self.per_stock_stats = {}
            for tid, group in df.groupby("ticker_id"):
                vals = group[self.FEATURE_COLS].values.astype(np.float32)
                mean = vals.mean(axis=0)
                std = vals.std(axis=0)
                std[std < 1e-6] = 1.0
                self.per_stock_stats[int(tid)] = {"mean": mean.tolist(), "std": std.tolist()}
        else:
            self.per_stock_stats = per_stock_stats

        # Build windows per stock
        for tid, group in df.groupby("ticker_id"):
            group = group.sort_values("time_idx").reset_index(drop=True)
            tid_int = int(tid)

            # Get this stock's own normalization stats
            if tid_int in self.per_stock_stats:
                s = self.per_stock_stats[tid_int]
                mean = np.array(s["mean"], dtype=np.float32)
                std = np.array(s["std"], dtype=np.float32)
            else:
                vals = group[self.FEATURE_COLS].values.astype(np.float32)
                mean = vals.mean(axis=0)
                std = vals.std(axis=0)
                std[std < 1e-6] = 1.0

            raw = group[self.FEATURE_COLS].values.astype(np.float32)
            normed = (raw - mean) / std
            targets = group["direction_7d"].astype(int).values

            n = len(group)
            for start in range(n - window_size):
                end = start + window_size
                x = normed[start:end]
                y = targets[end - 1]
                if y in (0, 1):
                    self.samples.append(
                        (
                            torch.tensor(x, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.long),
                        )
                    )

        print(f"  {len(self.samples):,} samples from {df['ticker_id'].nunique()} stocks")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — ENHANCED CNN-LSTM MODEL
#
# FIX 5: 3rd CNN layer with kernel=10 captures longer-term patterns.
# FIX 6: Residual connection around LSTM improves gradient flow.
# ══════════════════════════════════════════════════════════════════════════


class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        num_features=12,
        window_size=60,
        num_filters=64,
        num_filters2=128,
        num_filters3=128,
        kernel1=5,
        kernel2=3,
        kernel3=10,
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden

        # CNN block 1: short-term (1 week)
        self.conv1 = nn.Conv1d(num_features, num_filters, kernel1)
        self.bn1 = nn.BatchNorm1d(num_filters)

        # CNN block 2: pattern combinations
        self.conv2 = nn.Conv1d(num_filters, num_filters2, kernel2)
        self.bn2 = nn.BatchNorm1d(num_filters2)

        # CNN block 3: 2-week patterns — FIX 5
        self.conv3 = nn.Conv1d(num_filters2, num_filters3, kernel3)
        self.bn3 = nn.BatchNorm1d(num_filters3)

        # Pool
        self.pool = nn.MaxPool1d(2)

        # Projection: 128 -> lstm_hidden for residual match — FIX 6
        self.proj = nn.Linear(num_filters3, lstm_hidden)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, 60, 12]
        x = x.transpose(1, 2)  # [B, 12, 60]
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 128, 54]
        x = self.relu(self.bn3(self.conv3(x)))  # [B, 128, 45]
        x = self.pool(x)  # [B, 128, 22]
        x = x.transpose(1, 2)  # [B, 22, 128]

        # Project and apply residual — FIX 6
        x_proj = self.proj(x)  # [B, 22, 64]
        lstm_out, _ = self.lstm(x_proj)  # [B, 22, 64]
        x = lstm_out + x_proj  # residual: [B, 22, 64]

        x = x[:, -1, :]  # [B, 64]
        x = self.dropout(x)
        return self.fc(x)  # [B, 2]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df["direction_7d"] = df["direction_7d"].astype(int)
df["ticker_id"] = df["ticker_id"].astype(int)
df["company_type_id"] = df["company_type_id"].astype(int)

print(f"Loaded: {len(df):,} rows, {df['ticker_id'].nunique()} stocks")

if SMOKE_TEST:
    smoke_ids = sorted(df["ticker_id"].unique())[:SMOKE_STOCKS]
    df = df[df["ticker_id"].isin(smoke_ids)]
    print(f"Smoke: {df['ticker'].unique().tolist()}, rows={len(df):,}")

n_total = len(df)
n_up = (df["direction_7d"] == 1).sum()
n_down = (df["direction_7d"] == 0).sum()
up_pct = n_up / n_total * 100
print(f"Class balance: UP={up_pct:.1f}%  DOWN={100-up_pct:.1f}%")

# FIX 3: Correct class weight formula
# weight[c] = total / (num_classes * count[c])
# Rare class -> small count -> large weight -> model pays more attention
w_down = n_total / (2.0 * n_down)
w_up = n_total / (2.0 * n_up)
class_weights = torch.tensor([w_down, w_up], dtype=torch.float32).to(DEVICE)
print(f"Corrected weights: DOWN={w_down:.3f}, UP={w_up:.3f}")

TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"
train_df = df[df["date"] < TRAIN_END].copy()
val_df = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].copy()
test_df = df[df["date"] >= VAL_END].copy()
print(f"Train:{len(train_df):,} Val:{len(val_df):,} Test:{len(test_df):,}")

print("\nBuilding per-stock normalized datasets...")
print("  Training:")
train_ds = BVMTWindowDataset(train_df, WINDOW_SIZE, fit_scaler=True)
print("  Validation:")
val_ds = BVMTWindowDataset(
    val_df,
    WINDOW_SIZE,
    fit_scaler=False,
    per_stock_stats=train_ds.per_stock_stats,
)
print("  Test:")
test_ds = BVMTWindowDataset(
    test_df,
    WINDOW_SIZE,
    fit_scaler=False,
    per_stock_stats=train_ds.per_stock_stats,
)

# Save per-stock scaler stats
sp = RESULTS_DIR / "cnn_lstm_v2_per_stock_stats.json"
with open(sp, "w") as f:
    json.dump({str(k): v for k, v in train_ds.per_stock_stats.items()}, f)
print(f"Per-stock stats saved: {sp}")

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Batches — train:{len(train_loader)} val:{len(val_loader)}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — MODEL + OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════

print("\nBuilding CNN-LSTM v2...")
model = CNNLSTMModel(
    num_features=12,
    window_size=WINDOW_SIZE,
    num_filters=NUM_FILTERS,
    num_filters2=NUM_FILTERS2,
    num_filters3=NUM_FILTERS3,
    kernel1=KERNEL_SIZE1,
    kernel2=KERNEL_SIZE2,
    kernel3=KERNEL_SIZE3,
    lstm_hidden=LSTM_HIDDEN,
    lstm_layers=LSTM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

print(f"Parameters: {model.count_parameters():,}")
print(f"Architecture: CNN(k=5→k=3→k=10) + ResLSTM({LSTM_HIDDEN}x{LSTM_LAYERS})")

criterion = nn.CrossEntropyLoss(weight=class_weights)

# FIX 2: AdamW with lower LR
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# Warmup scheduler: linearly increase LR for first WARMUP_EPOCHS
def lr_warmup(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    return 1.0


warmup_sched = optim.lr_scheduler.LambdaLR(optimizer, lr_warmup)

plateau_kwargs = {
    "mode": "min",
    "factor": 0.5,
    "patience": 4,
}
if "verbose" in inspect.signature(optim.lr_scheduler.ReduceLROnPlateau).parameters:
    plateau_kwargs["verbose"] = True

plateau_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **plateau_kwargs)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRAIN / VALIDATE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════


def train_one_epoch(model, loader, criterion, optimizer, device, ep, total):
    model.train()
    tot_loss = 0.0
    correct = 0
    n = 0
    nb = len(loader)

    for bi, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)

        if bi % 100 == 0 or bi == nb:
            print(
                f"\r  E{ep:02d}/{total} [{bi}/{nb}] "
                f"loss={tot_loss/n:.4f} acc={correct/n*100:.1f}%",
                end="",
                flush=True,
            )
    print()
    return tot_loss / n, correct / n * 100



def run_eval(model, loader, criterion, device):
    model.eval()
    tot_loss = c = n = cup = cdn = nup = ndn = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            tot_loss += loss.item() * len(y)

            preds = out.argmax(1)
            c += (preds == y).sum().item()
            n += len(y)

            um = y == 1
            dm = y == 0
            cup += (preds[um] == 1).sum().item()
            nup += um.sum().item()
            cdn += (preds[dm] == 0).sum().item()
            ndn += dm.sum().item()

    acc_up = cup / max(nup, 1) * 100
    acc_down = cdn / max(ndn, 1) * 100
    collapsed = abs(acc_up - acc_down) > 30
    return tot_loss / n, c / n * 100, acc_up, acc_down, collapsed


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
seed_tag = f"_seed{SEED}" if SEED is not None else ""
run_name = f"{'smoke' if SMOKE_TEST else 'full'}_cnnlstm_v2_{timestamp}{seed_tag}"
mpath = MODEL_DIR / f"cnn_lstm_v2_{run_name}_best.pt"

best_loss = float("inf")
best_acc = 0.0
pat_count = 0
PATIENCE = 8
history = []

print("\n" + "=" * 60)
print("CNN-LSTM v2 TRAINING — fixes applied:")
print("  Per-stock normalization | LR=0.0003 + warmup")
print("  Correct class weights   | 3rd CNN layer (k=10)")
print("  Residual LSTM connection")
print("=" * 60)
print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

for ep in range(1, MAX_EPOCHS + 1):
    tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, ep, MAX_EPOCHS)
    vl, va, vup, vdn, col = run_eval(model, val_loader, criterion, DEVICE)

    # Step schedulers
    if ep <= WARMUP_EPOCHS:
        warmup_sched.step()
    else:
        plateau_sched.step(vl)

    lr = optimizer.param_groups[0]["lr"]
    imp = vl < best_loss
    tag = " <- NEW BEST" if imp else ""
    col_warn = " [COLLAPSE!]" if col else ""

    print(
        f"Epoch {ep:2d}/{MAX_EPOCHS} | "
        f"train loss={tl:.4f} acc={ta:.1f}% | "
        f"val loss={vl:.4f} acc={va:.1f}% "
        f"(UP={vup:.0f}% DN={vdn:.0f}%)"
        f"{tag}{col_warn} | lr={lr:.6f}"
    )

    history.append(
        {
            "epoch": ep,
            "train_loss": round(tl, 4),
            "train_acc": round(ta, 2),
            "val_loss": round(vl, 4),
            "val_acc": round(va, 2),
            "acc_up": round(vup, 1),
            "acc_down": round(vdn, 1),
            "lr": lr,
        }
    )

    if imp:
        best_loss = vl
        best_acc = va
        pat_count = 0
        torch.save(
            {
                "epoch": ep,
                "model_state": model.state_dict(),
                "val_loss": vl,
                "val_acc": va,
                "acc_up": vup,
                "acc_down": vdn,
                "model_config": {
                    "num_features": 12,
                    "window_size": WINDOW_SIZE,
                    "num_filters": NUM_FILTERS,
                    "num_filters2": NUM_FILTERS2,
                    "num_filters3": NUM_FILTERS3,
                    "kernel1": KERNEL_SIZE1,
                    "kernel2": KERNEL_SIZE2,
                    "kernel3": KERNEL_SIZE3,
                    "lstm_hidden": LSTM_HIDDEN,
                    "lstm_layers": LSTM_LAYERS,
                    "dropout": DROPOUT,
                },
                "per_stock_stats": {str(k): v for k, v in train_ds.per_stock_stats.items()},
            },
            mpath,
        )
    else:
        pat_count += 1

    if pat_count >= PATIENCE:
        print(
            f"\n[Early stop] {PATIENCE} epochs no improvement. "
            f"Best epoch {ep-PATIENCE}: val_loss={best_loss:.4f}"
        )
        break

print(f"\nFinished: {datetime.now():%Y-%m-%d %H:%M:%S}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST EVALUATION (2025 — never seen during training)")
print("=" * 60)

ck = torch.load(mpath, map_location=DEVICE)
model.load_state_dict(ck["model_state"])
print(
    f"Loaded: epoch {ck['epoch']}, val_loss={ck['val_loss']:.4f}, "
    f"UP={ck['acc_up']:.0f}%, DN={ck['acc_down']:.0f}%"
)

tl2, ta2, tup, tdn, _ = run_eval(model, test_loader, criterion, DEVICE)

interp = "STRONG" if ta2 >= 58 else "ACCEPTABLE" if ta2 >= 54 else "MARGINAL" if ta2 >= 50 else "WEAK"

print(f"\n  test_loss     = {tl2:.4f}")
print(f"  test_accuracy = {ta2:.1f}%  [{interp}]")
print(f"  UP accuracy   = {tup:.1f}%")
print(f"  DOWN accuracy = {tdn:.1f}%")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 — SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════

res = {
    "model": "CNN-LSTM-v2",
    "run_name": run_name,
    "seed": SEED,
    "smoke_test": SMOKE_TEST,
    "best_val_loss": round(best_loss, 4),
    "best_val_acc": round(best_acc, 2),
    "test_loss": round(tl2, 4),
    "test_acc": round(ta2, 2),
    "test_up_acc": round(tup, 1),
    "test_down_acc": round(tdn, 1),
    "total_params": model.count_parameters(),
    "fixes_applied": [
        "per_stock_normalization",
        "lr_warmup_0.0003",
        "correct_class_weights",
        "cnn_layer3_kernel10",
        "residual_lstm_connection",
    ],
    "history": history,
}

rp = RESULTS_DIR / f"results_cnnlstm_v2_{run_name}.json"
with open(rp, "w") as f:
    json.dump(res, f, indent=2)

print()
print("=" * 60)
print("COMPLETE — CNN-LSTM v2")
print("=" * 60)
print(f"  Best model : {mpath}")
print(f"  Results    : {rp}")
print()
print("THESIS TABLE (Chapter 5, Table 5.1):")
print(f"  CNN-LSTM v2 val_loss = {best_loss:.4f}")
print(f"  CNN-LSTM v2 val_acc  = {best_acc:.1f}%")
print(f"  CNN-LSTM v2 test_acc = {ta2:.1f}%")
print("=" * 60)
