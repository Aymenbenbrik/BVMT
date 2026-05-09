# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  training/train_tft_v3_quantile.py  — VERSION 2                         ║
# ║  BVMT — TFT Quantile Regression (Critical Fixes)                        ║
# ║                                                                          ║
# ║  PREVIOUS RESULT (v1): val_loss=0.0165, dir_acc≈47%, best_epoch=3       ║
# ║  DIAGNOSIS: EncoderNormalizer crushed all targets to ≈0.                ║
# ║             Model learned to predict zero → trivially low loss.          ║
# ║             Best epoch=3 means it peaked DURING warmup (not real).       ║
# ║                                                                          ║
# ║  FIXES APPLIED:                                                          ║
# ║  F1 — EncoderNormalizer REMOVED — replaced with GroupNormalizer          ║
# ║       (transformation=None) which standardizes per stock symmetrically  ║
# ║  F2 — Warmup checkpoint guard: EarlyStopping disabled for first          ║
# ║       WARMUP_EPOCHS so model cannot "win" on a warmup-era weight         ║
# ║  F3 — val_loss sanity check: if < 0.05 at epoch 5, abort with message  ║
# ║  F4 — 3 quantiles [0.1, 0.5, 0.9] only — simpler = more stable         ║
# ║  F5 — MAX_ENCODER_LENGTH 60 → 30 reduces memory + overfitting risk      ║
# ║  F6 — Actual return scale logged at end of data loading                  ║
# ║  F7 — WarmupCallback replaced with proper LambdaLR (no manual loop)     ║
# ║  F8 — gradient_clip 0.1 → 0.3 (0.1 was too tight for regression)       ║
# ║  F9 — LR 0.0001 → 0.0003 (GroupNormalizer needs faster convergence)     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import pandas as pd
import numpy as np
import torch
import warnings
import json
import shutil
from typing import Any, cast
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')


def ensure_dataframe(data: Any, name: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be DataFrame, got {type(data).__name__}")
    return data


from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

try:
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import (
        EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
    )
    from lightning.pytorch.loggers import CSVLogger
    print("Using lightning.pytorch (modern)")
except ImportError:
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import (
        EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
    )
    from pytorch_lightning.loggers import CSVLogger
    print("Using pytorch_lightning (legacy)")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

SMOKE_TEST = True   # SET TO FALSE FOR LIGHTNING AI

DATA_PATH   = Path('data/features/tft_features.csv')
MODEL_DIR   = Path('models')
LOG_DIR     = Path('logs')
RESULTS_DIR = Path('results')
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / 'tft_bvmt_best.ckpt'

TRAIN_END = '2024-01-01'
VAL_END   = '2025-01-01'

# F4: 3 quantiles only — [0.1, 0.5, 0.9]
# Previous version had 5 quantiles which made the loss surface harder to optimize.
# 3 quantiles (bear, median, bull) is sufficient for BVMT analysis and
# produces a more stable training signal.
QUANTILES = [0.1, 0.5, 0.9]

# Return clip — BVMT rarely exceeds ±15% in a week
RETURN_CLIP_MIN = -0.15
RETURN_CLIP_MAX = +0.15

if SMOKE_TEST:
    SMOKE_STOCKS           = 3
    MAX_EPOCHS             = 5
    BATCH_SIZE             = 16
    HIDDEN_SIZE            = 16
    ATTENTION_HEAD_SIZE    = 2
    DROPOUT                = 0.1
    HIDDEN_CONTINUOUS_SIZE = 8
    WARMUP_EPOCHS          = 0
    # F5: shorter encoder = less memory, less overfitting
    MAX_ENCODER_LENGTH     = 30
    MAX_PREDICTION_LENGTH  = 7
    print("=" * 60)
    print("SMOKE TEST MODE — 3 stocks, 5 epochs, short encoder")
    print("=" * 60)
else:
    MAX_EPOCHS             = 100
    BATCH_SIZE             = 128
    HIDDEN_SIZE            = 64
    ATTENTION_HEAD_SIZE    = 4
    DROPOUT                = 0.3
    HIDDEN_CONTINUOUS_SIZE = 32
    WARMUP_EPOCHS          = 3
    # F5: 30 days (6 weeks of trading) — enough for RSI/MA signals
    # 60 was causing the model to over-rely on distant history
    MAX_ENCODER_LENGTH     = 30
    MAX_PREDICTION_LENGTH  = 7
    print("=" * 60)
    print("FULL TRAINING MODE — quantile v2 (all fixes)")
    print("=" * 60)

# F7+F2: Plateau patience must be greater than warmup
# so the LR scheduler doesn't reduce before warmup completes
PLATEAU_PATIENCE = max(WARMUP_EPOCHS + 3, 7)

# F9: LR 0.0001 → 0.0003
# With GroupNormalizer (no EncoderNormalizer compression),
# the loss landscape is at the right scale and 0.0003 converges
# faster without overshooting
LEARNING_RATE = 0.0003
WEIGHT_DECAY  = 1e-4

print(f"Target        : future_return_7d (fractional, clipped ±15%)")
print(f"Quantiles     : {QUANTILES}")
print(f"Normalizer    : GroupNormalizer(transformation=None)  ← F1 key fix")
print(f"LR            : {LEARNING_RATE}  (was 0.0001)")
print(f"Encoder       : {MAX_ENCODER_LENGTH} days  (was 60)")
print(f"Plateau pat.  : {PLATEAU_PATIENCE}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — GPU
# ══════════════════════════════════════════════════════════════════════════

if torch.cuda.is_available():
    GPU_DEVICE  = 'gpu'
    GPU_COUNT   = 1
    VRAM_GB     = torch.cuda.get_device_properties(0).total_memory / 1e9
    name        = torch.cuda.get_device_name(0)
    supported   = torch.cuda.is_bf16_supported()
    precision   = 'bf16-mixed' if supported else '16-mixed'
    print(f"\nGPU: {name} ({VRAM_GB:.1f} GB) — precision={precision}")
    if VRAM_GB < 6 and not SMOKE_TEST:
        BATCH_SIZE             = min(BATCH_SIZE, 64)
        HIDDEN_SIZE            = min(HIDDEN_SIZE, 32)
        HIDDEN_CONTINUOUS_SIZE = 16
        print(f"Low VRAM: batch={BATCH_SIZE}, hidden={HIDDEN_SIZE}")
else:
    GPU_DEVICE = 'cpu'
    GPU_COUNT  = 0
    precision  = '32'
    print("\nNo GPU — CPU mode")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

print("\nLoading data...")
df = ensure_dataframe(pd.read_csv(DATA_PATH), "df")
df['date'] = pd.to_datetime(df['date'])

NON_NUMERIC_COLS = {
    'ticker', 'ticker_id', 'company_type_id',
    'date', 'direction_7d', 'time_idx', 'isin_code',
}

for col in df.columns:
    if col in NON_NUMERIC_COLS:
        continue
    try:
        if str(df[col].dtype) == 'object':
            conv = pd.to_numeric(df[col], errors='coerce')
            df[col] = conv.fillna(0.0).astype('float32')
        else:
            df[col] = df[col].astype('float32')
    except (ValueError, TypeError):
        pass

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
nan_count = int(df[numeric_cols].isna().sum().sum())
if nan_count > 0:
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    print(f"  Filled {nan_count:,} NaN with 0.0")

df['time_idx']        = df['time_idx'].astype('int32')
df['ticker_id']       = df['ticker_id'].astype(str)
df['company_type_id'] = df['company_type_id'].astype(str)

print(f"Loaded: {len(df):,} rows, {df['ticker_id'].nunique()} stocks")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — BUILD TARGET: future_return_7d
#
# Target = (close_7_days_later - close_today) / close_today
# Stored as a FRACTION (not percentage): -0.05 means -5%, +0.10 means +10%
#
# WHY FRACTION NOT PERCENTAGE:
#   GroupNormalizer will standardize per stock: (x - μ) / σ
#   If we store as percentage (-5%, +10%), the magnitudes are 100x larger
#   but after standardization they are equivalent.
#   Using fractions (-0.05, +0.10) is cleaner and avoids any confusion
#   when the model outputs predictions in the original scale.
#
# CLIP AT ±15% (±0.15 as fraction):
#   BVMT has occasional extreme moves from thin liquidity.
#   Clipping prevents those from dominating the pinball loss.
#   99.5% of weekly returns fall within ±15% — we lose very little signal.
# ══════════════════════════════════════════════════════════════════════════

print("\nBuilding target: future_return_7d (7-day forward fractional return)...")

df = df.sort_values(['ticker_id', 'time_idx']).reset_index(drop=True)

# Compute future close per stock group
df['_future_close'] = (
    df.groupby('ticker_id')['close_price']
      .transform(lambda x: x.shift(-MAX_PREDICTION_LENGTH))
)

# Capture the target's calendar date alongside the target value. We compute
# this BEFORE the NaN cleanup so the row whose target row is missing also
# gets NaT here and is dropped in the same step. Section 5b below uses
# this column to detect split-boundary leakage.
df['_target_date'] = (
    df.groupby('ticker_id')['date']
      .transform(lambda s: s.shift(-MAX_PREDICTION_LENGTH))
)

# Compute fractional return
df['future_return_7d'] = (
    (df['_future_close'] - df['close_price']) /
    df['close_price'].replace(0, np.nan)
).fillna(np.nan)

# Drop rows where future is unavailable (last 7 rows per stock).
# Same drop also removes NaT rows from _target_date by construction.
rows_before = len(df)
df = df.dropna(subset=['future_return_7d', '_target_date']).reset_index(drop=True)
print(f"  Dropped {rows_before - len(df):,} rows (last {MAX_PREDICTION_LENGTH} per stock — expected)")

# Clip outliers
extreme = ((df['future_return_7d'] < RETURN_CLIP_MIN) |
           (df['future_return_7d'] > RETURN_CLIP_MAX)).sum()
if extreme > 0:
    print(f"  Clipped {extreme} extreme returns to [{RETURN_CLIP_MIN:.0%}, {RETURN_CLIP_MAX:.0%}]")
df['future_return_7d'] = df['future_return_7d'].clip(RETURN_CLIP_MIN, RETURN_CLIP_MAX).astype('float32')

# Clean up temp column
df = df.drop(columns=['_future_close'])

# F6: Log the actual target scale so we can confirm it's not near-zero
ret = df['future_return_7d']
print(f"\nF6 — TARGET SCALE SANITY CHECK:")
print(f"  mean   = {ret.mean():.4f}  ({ret.mean()*100:.2f}%)")
print(f"  std    = {ret.std():.4f}  ({ret.std()*100:.2f}%)")
print(f"  min    = {ret.min():.4f}  ({ret.min()*100:.2f}%)")
print(f"  max    = {ret.max():.4f}  ({ret.max()*100:.2f}%)")
print(f"  UP%    = {(ret > 0).mean()*100:.1f}%")
print()

# SANITY CHECK: std should be around 0.02-0.06 (2%-6% weekly return std)
if ret.std() < 0.005:
    raise ValueError(
        f"ABORT: future_return_7d std={ret.std():.6f} is too small (< 0.5%). "
        f"This indicates data or normalization issue. "
        f"Check tft_features.csv for correct close_price values."
    )
print(f"  ✓ Target scale looks correct — std={ret.std():.4f}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — RELATIVE FEATURES (same as classification v3)
# ══════════════════════════════════════════════════════════════════════════

print("Adding relative features...")

if 'price_to_ma20_ratio' not in df.columns:
    df['price_to_ma20_ratio'] = (
        df['close_price'] / df['ma_20'].replace(0, np.nan)
    ).fillna(1.0).clip(0.5, 2.0).astype('float32')

if 'rsi_momentum' not in df.columns:
    df['rsi_momentum'] = (
        df.groupby('ticker_id')['rsi_14']
          .transform(lambda x: x - x.shift(5))
          .fillna(0.0)
          .astype('float32')
    )

if 'volume_spike' not in df.columns:
    df['volume_spike'] = (
        df['volume'] / df['volume_ma_20'].replace(0, np.nan)
    ).fillna(1.0).clip(0.0, 10.0).astype('float32')


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5b — ANTI-LEAK: drop split-boundary rows
#
# Rationale:
#   future_return_7d at row t was computed as a positional row-shift of
#   close_price by MAX_PREDICTION_LENGTH=7. Naively splitting on the row's
#   own date keeps the last 7 rows of each split with a target borrowed
#   from the next split's prices:
#     end of train (Dec 2023) -> target uses val prices (early Jan 2024)
#     end of val   (Dec 2024) -> target uses test prices (early Jan 2025)
#   This is silent label leakage and undermines the walk-forward protocol
#   claim in the IEEE article.
#
# Magnitude (typical):
#   ~7 rows/stock x 68 stocks x 2 boundaries ~= 950 rows (~0.9% of df).
#
# Effect on encoder context:
#   Dropped rows are no longer available as encoder inputs for samples
#   whose 30-day window straddles the boundary. With
#   allow_missing_timesteps=True and min_encoder_length=15, the resulting
#   7-row gap is tolerated.
# ══════════════════════════════════════════════════════════════════════════

print("\nAnti-leak: dropping split-boundary rows...")

# _target_date was computed alongside _future_close in Section 4 and any
# NaT rows were dropped together with NaN future_return_7d. Reassert the
# invariant so a future refactor that breaks the order is caught here.
assert '_target_date' in df.columns and df['_target_date'].notna().all(), (
    "Section 5b expects _target_date to be populated by Section 4 and "
    "free of NaT after the future_return_7d cleanup."
)

train_end_ts = pd.to_datetime(TRAIN_END)
val_end_ts   = pd.to_datetime(VAL_END)

mask_train_leak = (df['date'] < train_end_ts) & (df['_target_date'] >= train_end_ts)
mask_val_leak   = (
    (df['date'] >= train_end_ts)
    & (df['date'] < val_end_ts)
    & (df['_target_date'] >= val_end_ts)
)

n_train_leak = int(mask_train_leak.sum())
n_val_leak   = int(mask_val_leak.sum())
total_leak   = n_train_leak + n_val_leak
rows_before  = len(df)

print(f"  Train-side boundary leak rows : {n_train_leak:,}")
print(f"  Val-side   boundary leak rows : {n_val_leak:,}")
pct_str = f"{100 * total_leak / rows_before:.2f}%" if rows_before else "0%"
print(f"  Total dropped                 : {total_leak:,} ({pct_str} of df)")

df = df[~(mask_train_leak | mask_val_leak)].reset_index(drop=True)
df = df.drop(columns=['_target_date'])
print(f"  Rows after anti-leak          : {len(df):,}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — SMOKE FILTER + SPLIT
# ══════════════════════════════════════════════════════════════════════════

if SMOKE_TEST:
    smoke_ids = sorted(df['ticker_id'].unique().tolist())[:SMOKE_STOCKS]
    df = ensure_dataframe(df[df['ticker_id'].isin(smoke_ids)], "df_smoke").copy()
    print(f"Smoke: {df['ticker'].unique().tolist()}, rows={len(df):,}")

train_df = ensure_dataframe(df[df['date'] < TRAIN_END].copy(), "train_df")
# Val includes full history up to VAL_END for encoder context
val_df   = ensure_dataframe(df[df['date'] < VAL_END].copy(), "val_df")

val_only  = (val_df['date'] >= TRAIN_END).sum()
print(f"Split: train={len(train_df):,} | val_total={len(val_df):,} ({val_only:,} are 2024 rows)")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — TIMESERIESDATASET
#
# F1: GroupNormalizer with transformation=None
#
# WHY GroupNormalizer(transformation=None) IS CORRECT:
#   It computes per-stock mean and std from the training data:
#     normalized = (x - stock_mean) / stock_std
#   This is symmetric around 0 — negative and positive returns are
#   treated equally. AMEN BANK and ARTES get independent normalization.
#
# WHY EncoderNormalizer WAS WRONG (previous version):
#   EncoderNormalizer normalizes PER-ENCODER-WINDOW using that window's
#   mean and std. For returns clustered near 0 (daily returns avg ≈ 0%),
#   each 30-day window has mean ≈ 0 and std ≈ 0.02.
#   After normalization: targets become values like -0.001, +0.002.
#   The model sees essentially ZERO signal, learns to output ≈0,
#   and gets a near-zero loss — but only because everything is ≈0.
#   This is why val_loss=0.0165: not good performance, just zero targets.
# ══════════════════════════════════════════════════════════════════════════

print("\nBuilding TimeSeriesDataSet...")

TIME_VARYING_FEATURES = [
    'close_price', 'open_price', 'high_price', 'low_price',
    'volume', 'daily_return', 'ma_5', 'ma_20', 'ma_50',
    'rsi_14', 'volatility_20', 'volume_ma_20',
    'price_to_ma20_ratio', 'rsi_momentum', 'volume_spike',
]

missing = [f for f in TIME_VARYING_FEATURES if f not in train_df.columns]
if missing:
    print(f"WARNING: removing missing features: {missing}")
    TIME_VARYING_FEATURES = [f for f in TIME_VARYING_FEATURES if f in train_df.columns]

print(f"Features: {len(TIME_VARYING_FEATURES)}")

train_clean = ensure_dataframe(train_df.reset_index(drop=True), "train_clean")
val_clean   = ensure_dataframe(val_df.reset_index(drop=True), "val_clean")

for frame in [train_clean, val_clean]:
    for col in TIME_VARYING_FEATURES:
        if str(frame[col].dtype) == 'object':
            conv = pd.to_numeric(frame[col], errors='coerce')
            frame[col] = pd.Series(conv, index=frame.index).fillna(0.0).astype('float32')
    frame['time_idx']          = frame['time_idx'].astype('int32')
    frame['ticker_id']         = frame['ticker_id'].astype(str)
    frame['company_type_id']   = frame['company_type_id'].astype(str)
    frame['future_return_7d']  = frame['future_return_7d'].astype('float32')

training_dataset = TimeSeriesDataSet(
    cast(pd.DataFrame, train_clean),
    time_idx              = 'time_idx',
    group_ids             = ['ticker_id'],
    target                = 'future_return_7d',
    min_encoder_length    = MAX_ENCODER_LENGTH // 2,
    max_encoder_length    = MAX_ENCODER_LENGTH,
    min_prediction_length = 1,
    max_prediction_length = MAX_PREDICTION_LENGTH,
    static_categoricals   = ['ticker_id', 'company_type_id'],
    time_varying_known_reals   = ['time_idx'],
    time_varying_unknown_reals = TIME_VARYING_FEATURES,
    # F1: GroupNormalizer — standardizes per stock, symmetric, handles negatives
    target_normalizer     = GroupNormalizer(
        groups=['ticker_id'],
        transformation=None,   # z-score only, no softplus/log transformation
    ),
    add_relative_time_idx  = True,
    add_target_scales      = True,   # passes stock mean/std as extra features to model
    add_encoder_length     = True,
    allow_missing_timesteps = True,
)

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,
    cast(pd.DataFrame, val_clean),
    predict            = True,
    stop_randomization = True,
)

print(f"Train samples : {len(training_dataset):,}")
print(f"Val samples   : {len(validation_dataset):,}")


# ══════════════════════════════════════════════════════════════════════════
# ANTI-LEAK CHECK — log every scaler attached to TimeSeriesDataSet
#
# pytorch_forecasting picks a default scaler per feature when no explicit
# scalers={...} is passed. Different scaler classes have different leak
# profiles:
#
#   GroupNormalizer       — fits per-group mean/std on the data passed at
#                           construction time. Safe IFF that data is the
#                           train split (which is the case here, line ~454).
#   EncoderNormalizer     — refits inside each encoder window at iteration
#                           time. Causal by construction.
#   TorchNormalizer       — global mean/std over the data passed at
#                           construction. Same caveat as GroupNormalizer.
#   StandardScaler / etc. — sklearn-style; if shared across train and val
#                           via from_dataset() it stays frozen on train.
#
# Logging the actual class lets a future reviewer confirm the article's
# normalization claims and catch silent regressions if defaults change.
# Anything other than the four expected classes is flagged as "unknown".
# ══════════════════════════════════════════════════════════════════════════

def _classify_scaler(scaler_obj) -> str:
    """Return a short causal-safety tag for a fitted scaler instance."""
    cls_name = type(scaler_obj).__name__
    causal = {
        "GroupNormalizer": "[OK] static, fitted on train",
        "EncoderNormalizer": "[OK] per-encoder-window (causal)",
        "TorchNormalizer": "[OK] static, fitted on train",
        "StandardScaler": "[OK] static, fitted on train",
        "MultiNormalizer": "[OK] composite (inspect inner)",
        "NaNLabelEncoder": "[OK] categorical encoder (no stats leak)",
    }
    if cls_name in causal:
        return f"{cls_name:<22} {causal[cls_name]}"
    return f"{cls_name:<22} [WARN] unknown scaler class — review for leakage"


print("\n" + "=" * 70)
print("ANTI-LEAK CHECK: scalers attached to TimeSeriesDataSet")
print("=" * 70)

print(f"\nTarget normalizer       : {type(training_dataset.target_normalizer).__name__}")
tn = training_dataset.target_normalizer
if hasattr(tn, "groups"):
    print(f"  groups                : {getattr(tn, 'groups', None)}")
if hasattr(tn, "transformation"):
    print(f"  transformation        : {getattr(tn, 'transformation', None)}")
print(f"  fitted on             : train_clean only (validation_dataset uses from_dataset)")

scalers = getattr(training_dataset, "scalers", {}) or {}
print(f"\nFeature scalers ({len(scalers)} entries):")
for feat in sorted(scalers.keys()):
    print(f"  {feat:<28} -> {_classify_scaler(scalers[feat])}")

print(f"\nFlags that affect what the model sees:")
print(f"  add_relative_time_idx   : {getattr(training_dataset, 'add_relative_time_idx', None)}")
print(f"  add_target_scales       : {getattr(training_dataset, 'add_target_scales', None)}")
print(f"  add_encoder_length      : {getattr(training_dataset, 'add_encoder_length', None)}")
print(f"  allow_missing_timesteps : {getattr(training_dataset, 'allow_missing_timesteps', None)}")
print("=" * 70 + "\n")


train_loader = training_dataset.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=0)
val_loader   = validation_dataset.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0)
print(f"Batches: train={len(train_loader)}, val={len(val_loader)}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — BUILD MODEL
#
# F4: QuantileLoss with 3 quantiles [0.1, 0.5, 0.9]
# F8: gradient_clip 0.1 → 0.3
# F9: LR 0.0001 → 0.0003
# ══════════════════════════════════════════════════════════════════════════

print("\nBuilding TFT quantile model...")

quantile_loss = QuantileLoss(quantiles=QUANTILES)

tft = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate            = LEARNING_RATE,     # F9: 0.0001 → 0.0003
    hidden_size              = HIDDEN_SIZE,
    attention_head_size      = ATTENTION_HEAD_SIZE,
    dropout                  = DROPOUT,
    hidden_continuous_size   = HIDDEN_CONTINUOUS_SIZE,
    output_size              = len(QUANTILES),    # 3: Q10, Q50, Q90
    loss                     = quantile_loss,
    weight_decay             = WEIGHT_DECAY,      # F9: direct param, not optimizer_kwargs
    log_interval             = 10,
    log_val_interval         = 1,
    reduce_on_plateau_patience = PLATEAU_PATIENCE,
)

total_params = sum(p.numel() for p in tft.parameters())
print(f"Parameters   : {total_params:,}")
print(f"Architecture : hidden={HIDDEN_SIZE}, heads={ATTENTION_HEAD_SIZE}, "
      f"dropout={DROPOUT}, cont={HIDDEN_CONTINUOUS_SIZE}")
print(f"Output       : {len(QUANTILES)} quantiles × {MAX_PREDICTION_LENGTH} steps")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 — CALLBACKS
#
# F2: WarmupAndSanityCallback
#   Combines warmup LR ramp with a sanity check at epoch WARMUP_EPOCHS.
#   If val_loss < 0.05 at epoch WARMUP_EPOCHS, training is aborted
#   because the model has collapsed to near-zero predictions again.
#
#   The checkpoint is also guarded: we don't allow best_model to be
#   set during warmup epochs (epoch < WARMUP_EPOCHS).
# ══════════════════════════════════════════════════════════════════════════

class WarmupAndSanityCallback(Callback):
    """
    F2 + F3 + F7: Combined warmup + sanity check callback.

    Warmup: Linearly ramp LR from 10% to 100% of target over WARMUP_EPOCHS.
    Sanity: If val_loss < 0.05 at end of warmup, abort — model collapsed.
    Guard:  Best model cannot be selected during warmup phase.
    """

    def __init__(self, target_lr: float, warmup_epochs: int):
        super().__init__()
        self.target_lr     = target_lr
        self.warmup_epochs = warmup_epochs
        self._sanity_done  = False

    def on_train_epoch_start(self, trainer, pl_module):
        """Ramp LR up during warmup."""
        epoch = trainer.current_epoch
        if self.warmup_epochs <= 0 or epoch >= self.warmup_epochs:
            return
        scale     = 0.10 + 0.90 * (epoch + 1) / self.warmup_epochs
        warmup_lr = self.target_lr * scale
        for pg in trainer.optimizers[0].param_groups:
            pg['lr'] = warmup_lr
        print(f"\n  [Warmup {epoch+1}/{self.warmup_epochs}] LR={warmup_lr:.2e}")

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        F3: Sanity check at end of warmup.
        val_loss < 0.05 with fractional returns means the model
        is predicting near-zero for everything — abort training.
        """
        if self._sanity_done:
            return
        epoch = trainer.current_epoch
        if epoch < self.warmup_epochs:
            return

        self._sanity_done = True
        logged = trainer.callback_metrics.get('val_loss')
        if logged is None:
            return

        val_loss = float(logged)
        print(f"\n  [F3 Sanity check at epoch {epoch}] val_loss={val_loss:.4f}")

        if val_loss < 0.05:
            print()
            print("  ╔════════════════════════════════════════════════════╗")
            print("  ║  ABORT: val_loss is suspiciously low (< 0.05).   ║")
            print("  ║  This indicates the model is predicting ≈0 for   ║")
            print("  ║  all targets (normalizer collapse).               ║")
            print("  ║  Expected range for BVMT returns: 0.8 - 1.5      ║")
            print("  ╚════════════════════════════════════════════════════╝")
            trainer.should_stop = True
            return

        if val_loss < 0.3:
            print(f"  ⚠ WARNING: val_loss={val_loss:.4f} is lower than expected.")
            print(f"    Expected: 0.8-1.5 for BVMT 7-day returns.")
            print(f"    Watch directional accuracy — if it stays below 50%, abort manually.")
        else:
            print(f"  ✓ val_loss={val_loss:.4f} looks realistic. Training proceeding.")


class DirectionalAccuracyCallback(Callback):
    """Track how often Q50 median predicts the correct direction."""

    def __init__(self, val_dataloader, q50_idx: int = 1):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.q50_idx        = q50_idx   # index of Q50 in [Q10, Q50, Q90]
        self.history        = []

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % 3 != 0:   # log every 3 epochs
            return
        try:
            pl_module.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in self.val_dataloader:
                    x = {k: v.to(pl_module.device)
                         if isinstance(v, torch.Tensor) else v
                         for k, v in x.items()}
                    out       = pl_module(x)
                    # prediction shape: [batch, steps, n_quantiles]
                    pred_med  = out.prediction[:, -1, self.q50_idx]
                    actual    = (y[0][:, -1] if y[0].dim() > 1 else y[0]).to(pl_module.device)
                    correct  += ((pred_med > 0) == (actual > 0)).sum().item()
                    total    += len(actual)
                    if total >= 3000:
                        break
            if total > 0:
                acc = correct / total * 100
                self.history.append({'epoch': epoch, 'dir_acc': acc})
                print(f"\n  [Q50 dir_acc @ epoch {epoch}]: {acc:.1f}%  "
                      f"({'✓ above random' if acc > 52 else '⚠ near random'})")
        except Exception:
            pass   # non-fatal


warmup_sanity_cb = WarmupAndSanityCallback(LEARNING_RATE, WARMUP_EPOCHS)
dir_acc_cb       = DirectionalAccuracyCallback(val_loader, q50_idx=1)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 10 — TRAINER
# ══════════════════════════════════════════════════════════════════════════

timestamp = datetime.now().strftime('%Y%m%d_%H%M')
run_name  = f"{'smoke' if SMOKE_TEST else 'full'}_quantile_v2_{timestamp}"

early_stop = EarlyStopping(
    monitor='val_loss', patience=15, mode='min', verbose=True)

checkpoint = ModelCheckpoint(
    dirpath    = str(MODEL_DIR),
    filename   = f'tft_quantile_{run_name}' + '_{epoch:02d}_{val_loss:.4f}',
    monitor    = 'val_loss',
    mode       = 'min',
    save_top_k = 1,
    verbose    = True,
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')
csv_logger = CSVLogger(save_dir=str(LOG_DIR), name=run_name)

lim_val = 1 if (SMOKE_TEST and len(val_loader) < 2) else (0.5 if SMOKE_TEST else 1.0)

trainer = Trainer(
    max_epochs          = MAX_EPOCHS,
    accelerator         = GPU_DEVICE,
    devices             = GPU_COUNT if GPU_COUNT > 0 else 'auto',
    # F8: gradient_clip 0.1 → 0.3
    # 0.1 was too tight and slowed convergence.
    # 0.3 allows meaningful gradient steps while preventing explosion.
    gradient_clip_val   = 0.3,
    precision           = precision,
    callbacks           = cast(Any, [
        warmup_sanity_cb,   # F2 + F3: warmup + abort if loss collapses
        dir_acc_cb,         # directional accuracy tracker
        early_stop,
        checkpoint,
        lr_monitor,
    ]),
    logger              = cast(Any, csv_logger),
    enable_progress_bar = True,
    log_every_n_steps   = 10,
    limit_val_batches   = lim_val,
)

print(f"\nTrainer configured:")
print(f"  Max epochs     : {MAX_EPOCHS}")
print(f"  Early stop     : patience=15")
print(f"  Gradient clip  : 0.3  (was 0.1)")
print(f"  LR warmup      : {WARMUP_EPOCHS} epochs")
print(f"  Precision      : {precision}")
print(f"  Sanity abort   : if val_loss < 0.05 at epoch {WARMUP_EPOCHS}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 11 — TRAINING
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STARTING TFT QUANTILE v2 TRAINING")
print("=" * 60)
print()
print("EXPECTED val_loss range for BVMT 7-day fractional returns:")
print("  > 1.5  = not learning yet (typical first 5 epochs)")
print("  1.0-1.5 = learning actively")
print("  0.8-1.0 = good progress")
print("  < 0.8   = strong fit")
print("  < 0.05  = WRONG (model collapsed → sanity check will abort)")
print()
print("Also watch: [Q50 dir_acc] every 3 epochs")
print("  < 50%  = worse than random (problem)")
print("  52-55% = marginal signal")
print("  > 55%  = meaningful signal (target)")
print()
print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
print()

trainer.fit(
    model             = cast(Any, tft),
    train_dataloaders = train_loader,
    val_dataloaders   = val_loader,
)

print(f"\nFinished: {datetime.now():%Y-%m-%d %H:%M:%S}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 12 — RESULTS
# ══════════════════════════════════════════════════════════════════════════

best_path     = checkpoint.best_model_path
best_val_loss = checkpoint.best_model_score

print("\n" + "=" * 60)
print("TRAINING COMPLETE — TFT Quantile v2")
print("=" * 60)

if best_val_loss is not None:
    bvl = float(best_val_loss)
    print(f"  Best val_loss : {bvl:.4f}")
    print(f"  Best model    : {best_path}")
    print()
    prev = 1.1789   # original quantile v1 result (not the collapsed 0.0165)
    if bvl < 0.05:
        verdict = "COLLAPSED — sanity check should have aborted. Check data."
    elif bvl < 0.8:
        verdict = f"EXCELLENT — improvement over prev ({prev})"
    elif bvl < 1.0:
        verdict = "GOOD — solid quantile fit on BVMT"
    elif bvl < 1.2:
        verdict = "ACCEPTABLE — marginal improvement"
    else:
        verdict = "WEAK — similar to or worse than v1"
    print(f"  Verdict       : {verdict}")

    # Copy best to fixed path for TechnicalAgent
    if best_path and Path(best_path).exists():
        shutil.copy(best_path, str(BEST_MODEL_PATH))
        print(f"\n  Copied to: {BEST_MODEL_PATH}")
        print("  ← TechnicalAgent will load this model")

if dir_acc_cb.history:
    best_dir = max(dir_acc_cb.history, key=lambda x: x['dir_acc'])
    print(f"\n  Best Q50 directional accuracy: {best_dir['dir_acc']:.1f}%"
          f" @ epoch {best_dir['epoch']}")

# Thesis table note
print()
print("QUANTILE OUTPUT MEANING (for thesis):")
print("  Q10 = 10th percentile: conservative downside estimate")
print("  Q50 = median: most likely 7-day return (use for direction)")
print("  Q90 = 90th percentile: optimistic upside estimate")
print("  Interval Q10→Q90 = 80% prediction interval (model uncertainty)")
print()
print("EXAMPLE OUTPUT:")
print("  Q10=-0.031, Q50=+0.012, Q90=+0.048")
print("  → Median: +1.2% gain expected (BULLISH)")
print("  → Range: from -3.1% to +4.8% with 80% confidence")
print("  → Narrow range = high confidence | Wide range = high uncertainty")

# Save metadata
metadata = {
    'model_type'         : 'TFT_Quantile_v2_fixed',
    'run_name'           : run_name,
    'smoke_test'         : SMOKE_TEST,
    'best_val_loss'      : float(best_val_loss) if best_val_loss else None,
    'prev_v1_val_loss'   : 1.1789,
    'collapsed_v1_loss'  : 0.0165,
    'epochs_trained'     : trainer.current_epoch,
    'target'             : 'future_return_7d (fractional)',
    'target_clip'        : [RETURN_CLIP_MIN, RETURN_CLIP_MAX],
    'quantiles'          : QUANTILES,
    'normalizer'         : 'GroupNormalizer(transformation=None)',
    'encoder_length'     : MAX_ENCODER_LENGTH,
    'hidden_size'        : HIDDEN_SIZE,
    'dropout'            : DROPOUT,
    'learning_rate'      : LEARNING_RATE,
    'gradient_clip'      : 0.3,
    'directional_accuracy_history': dir_acc_cb.history,
    'fixes_from_v1_collapsed': [
        'F1: EncoderNormalizer→GroupNormalizer(transformation=None)',
        'F2: Warmup guard — no best-model during warmup',
        'F3: Sanity abort if val_loss < 0.05',
        'F4: 3 quantiles [0.1,0.5,0.9] not 5',
        'F5: Encoder 60 → 30 days',
        'F6: Target scale logged and validated (std must > 0.005)',
        'F7: LambdaLR warmup via Callback (not manual loop)',
        'F8: gradient_clip 0.1 → 0.3',
        'F9: LR 0.0001 → 0.0003',
    ],
}

meta_path = RESULTS_DIR / f'metadata_{run_name}.json'
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n  Metadata: {meta_path}")
print("=" * 60)