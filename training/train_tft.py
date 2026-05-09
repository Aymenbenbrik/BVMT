# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  training/train_tft.py                                                   ║
# ║  BVMT — Temporal Fusion Transformer Training Script                     ║
# ║                                                                          ║
# ║  PURPOSE: Train a TFT model to predict 7-day stock direction for        ║
# ║  BVMT stocks. Uses the tft_features.csv produced by                     ║
# ║  feature_engineering.ipynb                                               ║
# ║                                                                          ║
# ║  WHERE TO RUN:                                                           ║
# ║  - SMOKE TEST: Run locally on your RTX 3050 (SMOKE_TEST = True)         ║
# ║  - FULL TRAINING: Run on Lightning AI T4 GPU (SMOKE_TEST = False)       ║
# ║                                                                          ║
# ║  HOW TO RUN:                                                             ║
# ║  python training/train_tft.py                                           ║
# ║                                                                          ║
# ║  EXPECTED DURATION:                                                      ║
# ║  Smoke test (3 stocks, 3 epochs): ~5-10 minutes on RTX 3050             ║
# ║  Full training (all stocks, 30 epochs): ~45-90 minutes on T4 GPU        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS
#
# WHAT EACH LIBRARY DOES:
#
# torch: PyTorch — the deep learning framework that TFT is built on.
#   Think of it as the engine that runs the neural network computations.
#   PyTorch handles all the math (matrix multiplications, gradients, etc.)
#   so you never have to implement it manually.
#
# pytorch_forecasting: A library built on top of PyTorch specifically for
#   time series forecasting. It provides the TFT model implementation,
#   the TimeSeriesDataSet class that handles sliding windows automatically,
#   and metrics like CrossEntropy for classification.
#
# pytorch_lightning: A framework that wraps PyTorch training loops.
#   Without it, you would have to write: for each epoch, for each batch,
#   forward pass, compute loss, backward pass, optimizer step, validation...
#   pytorch_lightning does all of this for you in the Trainer class.
#   You just call trainer.fit(model, train_dataloader, val_dataloader).
#
# pandas / numpy: Data loading and manipulation.
#
# pathlib: Clean file path handling on Windows.
#
# warnings: Suppress unimportant warnings that clutter the output.
# ══════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import torch
import warnings
import json
from typing import Any, cast
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')


def ensure_dataframe(data: Any, name: str) -> pd.DataFrame:
    """Type-safe guard for static analyzers and runtime sanity checks."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(data).__name__}")
    return data

# pytorch_forecasting provides the TFT model and data tools
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import CrossEntropy

# pytorch_lightning provides the training loop
# We try lightning.pytorch first (newer name), fall back to pytorch_lightning
# This fixes the "LightningModule is not assignable" linter warning on Lightning AI
try:
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        LearningRateMonitor,
    )
    from lightning.pytorch.loggers import CSVLogger
    print("Using lightning.pytorch (modern)")
except ImportError:
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        LearningRateMonitor,
    )
    from pytorch_lightning.loggers import CSVLogger
    print("Using pytorch_lightning (legacy)")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION
#
# All tunable settings are defined here in one place.
# Change values here, not scattered throughout the code.
#
# SMOKE_TEST MODE:
#   When True, trains on only 3 stocks for 3 epochs.
#   Purpose: confirm the entire pipeline works without errors before
#   committing to a 1-2 hour full training session on Lightning AI.
#   Always run smoke test first on your local GPU.
#   Expected smoke test output: loss decreases, no errors, .ckpt file saved.
#
# MAX_ENCODER_LENGTH (60 days):
#   How many past trading days TFT reads to make one prediction.
#   60 days = approximately 3 months of trading history.
#   Analogy: a financial analyst reviewing 3 months of charts before
#   making a buy/sell recommendation.
#   Why not more? More history = more memory needed. 60 is a good balance.
#   Why not less? Less than 30 days misses monthly patterns.
#
# MAX_PREDICTION_LENGTH (7 days):
#   How many days ahead TFT predicts.
#   We predict 7 trading days = approximately 1 calendar week.
#   This matches the direction_7d target computed in feature_engineering.
#
# BATCH_SIZE:
#   How many training examples are processed together in one forward pass.
#   Larger batch = faster training but more GPU memory required.
#   RTX 3050 (4GB): use 32 or 64
#   Lightning AI T4 (16GB): use 128 or 256
#
# LEARNING_RATE (0.001):
#   How fast the model updates its weights after each batch.
#   Too high (0.01): model overshoots, loss oscillates or increases
#   Too low (0.0001): model learns but very slowly
#   0.001 is the standard starting point for TFT.
#   The ReduceLROnPlateau scheduler will automatically reduce this
#   if validation loss stops improving.
#
# HIDDEN_SIZE (64):
#   The width of the neural network — how many neurons in each layer.
#   Larger = more capacity to learn complex patterns, but more memory.
#   RTX 3050: use 32
#   T4 GPU: use 64 or 128
#
# MAX_EPOCHS (30):
#   Maximum training passes through the full dataset.
#   In practice, EarlyStopping will stop training before epoch 30
#   if validation loss stops improving for 5 consecutive epochs.
# ══════════════════════════════════════════════════════════════════════════

# ── Toggle this to True for local smoke test, False for full training ──────
SMOKE_TEST = True  # SET TO FALSE BEFORE UPLOADING TO LIGHTNING AI

# ── File paths ─────────────────────────────────────────────────────────────
# Change these paths when running on Lightning AI
DATA_PATH    = Path('data/features/tft_features.csv')
MODEL_DIR    = Path('models')
LOG_DIR      = Path('logs')
RESULTS_DIR  = Path('results')

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── TFT architecture parameters ────────────────────────────────────────────
# Use the SMOKE_TEST values when testing locally, FULL values on Lightning AI
if SMOKE_TEST:
    SMOKE_STOCKS          = 3      # only train on 3 stocks
    MAX_EPOCHS            = 3      # only 3 epochs
    BATCH_SIZE            = 16     # small batch fits on any GPU
    HIDDEN_SIZE           = 16     # minimal model size
    ATTENTION_HEAD_SIZE   = 2
    DROPOUT               = 0.1
    HIDDEN_CONTINUOUS_SIZE = 8
    print("=" * 60)
    print("SMOKE TEST MODE — 3 stocks, 3 epochs")
    print("Purpose: confirm pipeline works before full training")
    print("=" * 60)
else:
    MAX_EPOCHS            = 30
    BATCH_SIZE            = 128    # reduce to 64 if out-of-memory on T4
    HIDDEN_SIZE           = 64     # increase to 128 if T4 allows
    ATTENTION_HEAD_SIZE   = 4
    DROPOUT               = 0.1
    HIDDEN_CONTINUOUS_SIZE = 32
    print("=" * 60)
    print("FULL TRAINING MODE — all stocks, up to 30 epochs")
    print("=" * 60)

# ── Shared parameters (same for both modes) ────────────────────────────────
MAX_ENCODER_LENGTH    = 60   # days of history TFT reads (3 months)
MAX_PREDICTION_LENGTH = 7    # days ahead TFT predicts (1 week)
LEARNING_RATE         = 0.001

# ── Train/Val/Test split dates ─────────────────────────────────────────────
# WHY THIS SPLIT STRATEGY (walk-forward validation):
#   We train on 2016-2023 (8 years), validate on 2024 (1 year),
#   and test on 2025 (1 year, never seen during training or tuning).
#   This mimics real-world usage: you train on the past, deploy for the future.
#   We never use random splits for time series — that would leak future data.
TRAIN_END   = '2024-01-01'   # training data: everything before this date
VAL_END     = '2025-01-01'   # validation data: 2024
# test data: 2025 (evaluated separately in evaluate_tft.py)

print(f"Train  : 2016-01-01 to {TRAIN_END}")
print(f"Val    : {TRAIN_END} to {VAL_END}")
print(f"Test   : {VAL_END} to end")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — GPU DETECTION
#
# WHAT THIS DOES:
#   Detects whether a GPU is available and configures the trainer accordingly.
#   PyTorch can run on CPU (slow) or GPU (fast).
#   Training on CPU: ~10x slower — acceptable for smoke test only.
#   Training on GPU: uses CUDA (NVIDIA's parallel computing platform).
#
# HOW GPU ACCELERATION WORKS (simple explanation):
#   A CPU has 4-16 powerful cores that process things sequentially.
#   A GPU has 3000-10000 small cores that process many things in parallel.
#   Neural network training is essentially matrix multiplication done
#   millions of times — perfect for parallel processing.
#   The RTX 3050 has 2560 CUDA cores. The T4 has 2560 cores but with
#   much more memory (16GB vs 4GB).
# ══════════════════════════════════════════════════════════════════════════

if torch.cuda.is_available():
    GPU_DEVICE   = 'gpu'
    GPU_COUNT    = 1
    DEVICE_NAME  = torch.cuda.get_device_name(0)
    VRAM_GB      = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\nGPU detected: {DEVICE_NAME}")
    print(f"VRAM available: {VRAM_GB:.1f} GB")

    # Auto-reduce batch size if VRAM is low (< 6 GB = RTX 3050)
    if VRAM_GB < 6 and not SMOKE_TEST:
        BATCH_SIZE = min(BATCH_SIZE, 64)
        HIDDEN_SIZE = min(HIDDEN_SIZE, 32)
        print(f"Low VRAM detected: reduced BATCH_SIZE to {BATCH_SIZE}, "
              f"HIDDEN_SIZE to {HIDDEN_SIZE}")
else:
    GPU_DEVICE  = 'cpu'
    GPU_COUNT   = 0
    print("\nNo GPU detected — running on CPU")
    print("Training will be slow. Use Lightning AI for full training.")
    if not SMOKE_TEST:
        print("WARNING: Full training on CPU takes 10+ hours. Switch to Lightning AI.")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATA LOADING AND DTYPE CLEANING
#
# THE FIX FOR: TypeError: can't convert np.ndarray of type numpy.object_
#
# ROOT CAUSE:
#   pandas loads some columns from CSV as dtype=object when the values
#   are mixed (e.g. integers and NaN together, or nullable Int64 from
#   PostgreSQL). PyTorch's as_tensor() cannot convert object arrays —
#   it only accepts float32, float64, int32, int64, bool, etc.
#
# THE FIX — 3 steps:
#   Step 1: Force every feature column to float32 using pd.to_numeric()
#           pd.to_numeric(errors='coerce') converts bad values to NaN
#           then we fill NaN with 0.0 so no nulls reach the model
#   Step 2: Fill all remaining NaN values in numeric columns with 0.0
#   Step 3: Verify no object dtype columns remain before building dataset
#
# WHY float32 and not float64?
#   float32 uses 4 bytes per number, float64 uses 8 bytes.
#   Neural networks train on float32 — using float64 doubles memory
#   usage with no accuracy benefit. All GPU tensor operations use float32.
#
# COLUMNS EXCLUDED FROM CASTING (legitimately non-numeric):
#   ticker, ticker_id, company_type_id, date, direction_7d
#   These will be cast to string separately for pytorch_forecasting.
# ══════════════════════════════════════════════════════════════════════════
 
# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATA LOADING AND DTYPE CLEANING
#
# FIXES APPLIED HERE:
#
# FIX 1 — object dtype → float32 (original error)
#   Some columns loaded as object due to PostgreSQL nullable integers.
#   pd.to_numeric(errors='coerce') converts them safely.
#
# FIX 2 — time_idx must stay integer
#   pytorch_forecasting requires int32/int64 for time_idx.
#   Our loop casts everything to float32 — we restore int32 after.
#
# FIX 3 — direction_7d: '0.0'/'1.0' vs '0'/'1'
#   If the CSV stored it as float (0.0, 1.0), direct astype(str)
#   gives '0.0' and '1.0' — pytorch_forecasting won't find '0'/'1'.
#   Safe chain: float → int → str gives exactly '0' and '1'.
#
# FIX 4 — NaN in direction_7d becomes the string 'nan'
#   astype(str) on NaN produces 'nan' — a third category.
#   CrossEntropy with 2 classes crashes if it sees 'nan' labels.
#   Fix: drop rows where direction_7d is 'nan' after casting.
#
# FIX 5 — ticker_id and company_type_id must never be float
#   pytorch_forecasting static_categoricals require string dtype.
#   We add them to NON_NUMERIC_COLS so the loop never touches them.
#   They get cast to string at the end of this section.
# ══════════════════════════════════════════════════════════════════════════

print("\nLoading data...")
df = ensure_dataframe(pd.read_csv(DATA_PATH), "df")
df['date'] = pd.to_datetime(df['date'])

# FIX 5: Include categorical columns in the exclusion set
# so the numeric casting loop never converts them to float32
NON_NUMERIC_COLS = {
    'ticker',           # stock name string e.g. 'AMEN BANK'
    'ticker_id',        # group identifier — must be string for pytorch_forecasting
    'company_type_id',  # static categorical — must be string
    'date',             # datetime — already parsed above
    'direction_7d',     # target — handled separately below
    'time_idx',         # integer index — handled separately below
}

print("Casting all feature columns to float32...")
fixed_cols = []

for col in df.columns:
    if col in NON_NUMERIC_COLS:
        continue  # skip — handled individually below

    col_dtype = str(df[col].dtype)

    if col_dtype == 'object':
        converted = pd.to_numeric(df[col], errors='coerce')
        if not isinstance(converted, pd.Series):
            converted = pd.Series(converted, index=df.index)
        df[col] = converted.astype('float32')
        fixed_cols.append(f"{col}(object→float32)")

    elif col_dtype in ('Int8','Int16','Int32','Int64',
                       'UInt8','UInt16','UInt32','UInt64'):
        df[col] = df[col].astype('float32')
        fixed_cols.append(f"{col}(Int64→float32)")

    else:
        try:
            df[col] = df[col].astype('float32')
        except (ValueError, TypeError):
            pass

# Fill NaN values introduced by coercion
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
nan_count    = df[numeric_cols].isna().sum().sum()
if nan_count > 0:
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    print(f"  Filled {nan_count:,} NaN values with 0.0")

if fixed_cols:
    print(f"  Fixed columns: {fixed_cols}")
else:
    print("  All feature columns already float32")

# Drop non-convertible columns that somehow slipped through
remaining_obj = [c for c in df.columns
                 if df[c].dtype == object and c not in NON_NUMERIC_COLS]
if remaining_obj:
    print(f"  Dropping non-convertible columns: {remaining_obj}")
    df = df.drop(columns=remaining_obj)
else:
    print("  Dtype check passed — all feature columns are numeric float32")

# FIX 2: Restore time_idx as integer
# The loop above would have cast it to float32 if it wasn't excluded.
# pytorch_forecasting validates: data[self.time_idx].dtype.kind == "i"
# kind "i" = signed integer. float32 has kind "f" → AssertionError.
df['time_idx'] = df['time_idx'].astype('int32')

# FIX 3 + FIX 4: Cast direction_7d safely to clean '0'/'1' strings
# Step 1: float→int removes the .0 decimal (0.0→0, 1.0→1)
# Step 2: int→str gives clean '0' and '1' (not '0.0' or '1.0')
# Step 3: drop rows where result is 'nan' (came from NaN target values)
df['direction_7d'] = (df['direction_7d']
                      .astype(float)       # handles '0', '0.0', 0, 0.0
                      .astype(int)         # removes decimal → 0 or 1
                      .astype(str))        # gives exactly '0' or '1'

rows_before = len(df)
df = df[df['direction_7d'].isin(['0', '1'])]
rows_dropped = rows_before - len(df)
if rows_dropped > 0:
    print(f"  Dropped {rows_dropped} rows with invalid direction_7d")

# FIX 5: Cast categorical columns to string for pytorch_forecasting
df['ticker_id']       = df['ticker_id'].astype(str)
df['company_type_id'] = df['company_type_id'].astype(str)

# Verify target values are exactly '0' and '1'
tgt_series = cast(pd.Series, df['direction_7d'])
unique_targets = sorted(tgt_series.unique().tolist())
print(f"  Target values: {unique_targets}")
# Must print: ['0', '1'] — anything else = problem

ticker_id_series = cast(pd.Series, df['ticker_id'])
print(f"Loaded: {len(df):,} rows, {ticker_id_series.nunique()} stocks")

# ── Smoke test filter ──────────────────────────────────────────────────────
if SMOKE_TEST:
    smoke_ids = sorted(cast(pd.Series, df['ticker_id']).unique().tolist())[:SMOKE_STOCKS]
    smoke_mask = cast(pd.Series, df['ticker_id']).isin(smoke_ids)
    df = ensure_dataframe(df[smoke_mask], "df_smoke").copy()
    smoke_names = cast(pd.Series, df['ticker']).dropna().unique().tolist()
    print(f"Smoke test: using {SMOKE_STOCKS} stocks: {smoke_names}")
    print(f"Rows after filter: {len(df):,}")

# ── Class balance — needed by Section 7 and Section 11 ───────────────────
up_pct   = (df['direction_7d'] == '1').mean()
down_pct = 1 - up_pct
print(f"\nClass balance: UP={up_pct*100:.1f}%  DOWN={down_pct*100:.1f}%")

if up_pct > 0.60:
    class_weights     = torch.tensor([up_pct, down_pct])
    USE_CLASS_WEIGHTS = True
    print(f"Class weights applied: DOWN={up_pct:.3f}, UP={down_pct:.3f}")
else:
    class_weights     = None
    USE_CLASS_WEIGHTS = False
    print("Class balance acceptable — no weighting needed")
# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAIN/VALIDATION SPLIT
#
# WHAT THIS DOES:
#   Splits the data into training (2016-2023) and validation (2024) sets.
#
# WHY WE CANNOT USE RANDOM SPLITS FOR TIME SERIES:
#   Imagine randomly splitting and getting:
#   - Training: March 2024 data
#   - Validation: January 2024 data
#   The model would be trained on future data and validated on past data.
#   It would appear to predict the future perfectly — but only because
#   it already saw the future during training. This is data leakage.
#
#   Walk-forward validation (what we use) is the academically correct
#   approach: always train on the past, validate on a future period.
#   Every real trading system is evaluated this way.
#
# WHY WE INCLUDE TRAIN DATA IN THE VALIDATION DATASET:
#   TimeSeriesDataSet.from_dataset() needs the training data to be
#   included with the validation data because TFT's encoder window
#   (60 days) needs to look back into training data even when
#   making predictions on 2024 validation dates.
#   Example: to predict January 3, 2024, TFT reads Oct-Dec 2023
#   (training data) as its 60-day encoder input.
#   Without training data included, there would be no history to read
#   for the first 60 days of validation.
# ══════════════════════════════════════════════════════════════════════════

train_df = ensure_dataframe(df[df['date'] < TRAIN_END].copy(), "train_df")
val_df   = ensure_dataframe(df[df['date'] < VAL_END].copy(), "val_df")

print(f"\nSplit sizes:")
print(f"  Training rows  : {len(train_df):,}  (2016 to {TRAIN_END})")
print(f"  Validation rows: {len(val_df[val_df['date'] >= TRAIN_END]):,}  (2024 only)")
print(f"  Val df total   : {len(val_df):,}  (includes train context for encoder)")
# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — TIMESERIESDATASET
#
# WHAT IS TimeSeriesDataSet?
#   It is pytorch_forecasting's data format for TFT.
#   It automatically handles the sliding window mechanism.
#
# WHAT IS A SLIDING WINDOW? (key concept)
#   Imagine a stock with 500 trading days.
#   TFT needs samples like:
#   Sample 1: days 1-60 (encoder input) → predict day 67 (day 60+7)
#   Sample 2: days 2-61 (encoder input) → predict day 68
#   Sample 3: days 3-62 (encoder input) → predict day 69
#   ... and so on until the last sample.
#
#   TimeSeriesDataSet creates all these sliding window samples
#   automatically from your flat CSV. Without it, you would have to
#   write this windowing logic manually — it is complex code.
#
# KEY PARAMETERS EXPLAINED:
#
# group_ids = ['ticker_id']:
#   Tells TimeSeriesDataSet that 'ticker_id' separates different
#   time series. It will create separate sliding windows for each stock.
#   Without this, it would try to slide across stock boundaries.
#
# target = 'direction_7d':
#   The column we want to predict. Must be a string for classification.
#
# min_encoder_length = MAX_ENCODER_LENGTH // 2:
#   Allow shorter encoder windows at the start of each stock's history.
#   Without this, the first 30 days of each stock would be unusable.
#
# static_categoricals = ['ticker_id', 'company_type_id']:
#   These columns are constant for each stock — TFT treats them as
#   static context embeddings that initialize the hidden state.
#
# time_varying_unknown_reals:
#   These are the historical features TFT reads through its LSTM encoder.
#   They are "unknown" because at prediction time we don't know their
#   future values — we can only look at their history.
#
# target_normalizer = NaNLabelEncoder():
#   direction_7d is categorical ('0'/'1'), so it must be label-encoded
#   to integer class IDs before conversion to torch tensors.
#   Setting target_normalizer=None would pass object labels directly and
#   cause TypeError: can't convert np.ndarray of type numpy.object_.
#
# add_relative_time_idx = True:
#   Adds a normalized time position (0 to 1) within each window.
#   Helps TFT understand "am I at the beginning or end of the window?"
# ══════════════════════════════════════════════════════════════════════════

print("\nBuilding TimeSeriesDataSet...")

TIME_VARYING_FEATURES = [
    'close_price', 'open_price', 'high_price', 'low_price',
    'volume', 'daily_return', 'ma_5', 'ma_20', 'ma_50',
    'rsi_14', 'volatility_20', 'volume_ma_20',
]

# Verify all features exist
missing = [f for f in TIME_VARYING_FEATURES if f not in train_df.columns]
if missing:
    print(f"WARNING: removing missing features: {missing}")
    TIME_VARYING_FEATURES = [f for f in TIME_VARYING_FEATURES
                              if f in train_df.columns]

# ── CRITICAL: reset_index() before passing to TimeSeriesDataSet ───────────
# pd.DataFrame() constructor and boolean slicing can silently revert
# column dtypes back to object. reset_index(drop=True) forces pandas
# to rebuild the DataFrame cleanly, preserving all dtype fixes.
# Do NOT use pd.DataFrame(train_df) — it defeats all our dtype work.
train_clean = ensure_dataframe(train_df.reset_index(drop=True), "train_clean")
val_clean   = ensure_dataframe(val_df.reset_index(drop=True), "val_clean")

# One final dtype enforcement on the clean frames
for frame in [train_clean, val_clean]:
    for col in TIME_VARYING_FEATURES:
        if frame[col].dtype == object:
            frame_col = pd.to_numeric(frame[col], errors='coerce')
            if not isinstance(frame_col, pd.Series):
                frame_col = pd.Series(frame_col, index=frame.index)
            frame[col] = frame_col.fillna(0.0).astype('float32')
    frame['time_idx']        = frame['time_idx'].astype('int32')
    frame['direction_7d']    = frame['direction_7d'].astype(str)
    frame['ticker_id']       = frame['ticker_id'].astype(str)
    frame['company_type_id'] = frame['company_type_id'].astype(str)

# Confirm no object dtype in feature columns
obj_cols = [c for c in TIME_VARYING_FEATURES
            if train_clean[c].dtype == object]
if obj_cols:
    print(f"ERROR: still object dtype: {obj_cols}")
else:
    print("  All feature columns confirmed float32")

training_dataset = TimeSeriesDataSet(
    cast(pd.DataFrame, train_clean),
    time_idx              = 'time_idx',
    group_ids             = ['ticker_id'],
    target                = 'direction_7d',
    min_encoder_length    = MAX_ENCODER_LENGTH // 2,
    max_encoder_length    = MAX_ENCODER_LENGTH,
    min_prediction_length = 1,
    max_prediction_length = MAX_PREDICTION_LENGTH,
    static_categoricals   = ['ticker_id', 'company_type_id'],
    time_varying_known_reals   = ['time_idx'],
    time_varying_unknown_reals = TIME_VARYING_FEATURES,

    target_normalizer     = NaNLabelEncoder(),
    add_relative_time_idx = True,  # adds position within window (0.0 to 1.0)
    add_target_scales     = False,
    add_encoder_length    = True,
    allow_missing_timesteps = True,
)

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,
    cast(pd.DataFrame, val_clean),
    predict            = True,
    stop_randomization = True,
)

print(f"Training dataset  : {len(training_dataset):,} samples")
print(f"Validation dataset: {len(validation_dataset):,} samples")

# ── Create DataLoaders ─────────────────────────────────────────────────────
# DataLoaders batch and shuffle the samples for efficient GPU training
# num_workers=0 is required on Windows (multiprocessing issues with >0)
train_dataloader = training_dataset.to_dataloader(
    train      = True,           # shuffle training data each epoch
    batch_size = BATCH_SIZE,
    num_workers = 0,             # Windows: always use 0
)
val_dataloader = validation_dataset.to_dataloader(
    train      = False,          # don't shuffle validation data
    batch_size = BATCH_SIZE,
    num_workers = 0,
)

print(f"\nDataLoaders ready")
print(f"  Train batches: {len(train_dataloader)}")
print(f"  Val batches  : {len(val_dataloader)}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — BUILD THE TFT MODEL
#
# WHAT IS TFT? (Temporal Fusion Transformer — explained simply)
#
# TFT is a neural network architecture designed specifically for
# multi-horizon time series forecasting. Published by Google Research in 2019.
# "Multi-horizon" means it can predict multiple steps ahead at once (days 1-7).
#
# TFT HAS 4 MAIN COMPONENTS:
#
# 1. Variable Selection Network (VSN):
#    Decides which features matter most for this particular stock.
#    EXAMPLE: For banks (AMEN BANK), RSI and MA20 might matter most.
#    For illiquid stocks, volume might matter more.
#    The VSN learns different feature importances for each company type.
#    This is what enables TFT to handle heterogeneous datasets.
#
# 2. LSTM Encoder (reads the past):
#    Reads the 60-day history window sequentially, like reading a sentence.
#    It maintains a "memory" that compresses 60 days of information into
#    a hidden state vector (the context vector).
#    ANALOGY: An analyst who reads 3 months of charts and mentally summarizes
#    the key trends before making a recommendation.
#
# 3. Multi-Head Attention (finds important patterns):
#    Looks back at all 60 days and decides which days were most informative.
#    EXAMPLE: "The day 3 weeks ago when volume spiked was important."
#    Multiple "heads" = multiple pattern detectors running in parallel.
#    Each head can find different types of patterns (trend, reversal, etc.)
#    THIS IS THE KEY INNOVATION of TFT vs regular LSTMs.
#
# 4. Point-wise Feed-forward Network (makes the final decision):
#    Takes the attended context and produces the final prediction.
#    For our task: "given everything above, predict UP (1) or DOWN (0)"
#
# KEY PARAMETERS:
#
# hidden_size (64):
#   The dimension of all internal representations.
#   A hidden_size=64 means each layer produces vectors of 64 numbers.
#   Larger = more capacity but more memory and slower training.
#
# attention_head_size (4):
#   Number of parallel attention patterns.
#   4 heads = 4 different pattern detectors scanning the 60-day window.
#   More heads = can find more types of patterns.
#
# dropout (0.1):
#   Randomly zeros out 10% of neurons during training.
#   This forces the model to not rely on any single neuron.
#   ANALOGY: Force a student to solve problems without their notes —
#   they learn more deeply and generalize better.
#   Without dropout: model memorizes training data (overfitting).
#
# loss = CrossEntropy:
#   The metric that measures how wrong each prediction is.
#   For classification (UP/DOWN), CrossEntropy is the standard choice.
#   A perfect prediction gets loss=0. A completely wrong one gets high loss.
#   During training, the model adjusts its weights to minimize this loss.
# ══════════════════════════════════════════════════════════════════════════

print("\nBuilding TFT model...")

# Build the loss function with optional class weighting
if USE_CLASS_WEIGHTS and class_weights is not None:
    loss_fn = CrossEntropy(weight=class_weights)
    print(f"Using weighted CrossEntropy: {class_weights.tolist()}")
else:
    loss_fn = CrossEntropy()
    print("Using standard CrossEntropy")

# from_dataset() reads the dataset structure and auto-configures the
# model dimensions, embedding sizes, etc. This is much easier than
# manually specifying every dimension.
tft = TemporalFusionTransformer.from_dataset(
    training_dataset,

    # ── Learning rate ──────────────────────────────────────────────────────
    learning_rate            = LEARNING_RATE,

    # ── Model architecture ────────────────────────────────────────────────
    hidden_size              = HIDDEN_SIZE,           # internal dimension
    attention_head_size      = ATTENTION_HEAD_SIZE,   # parallel attention patterns
    dropout                  = DROPOUT,               # regularization strength
    hidden_continuous_size   = HIDDEN_CONTINUOUS_SIZE,# size for continuous variable embeddings

    # ── Loss function ─────────────────────────────────────────────────────
    loss                     = loss_fn,

    # ── Logging ───────────────────────────────────────────────────────────
    log_interval             = 10,      # log metrics every 10 batches
    log_val_interval         = 1,       # log validation every epoch

    # ── Learning rate scheduling ──────────────────────────────────────────
    # If val_loss doesn't improve for 3 epochs, halve the learning rate
    reduce_on_plateau_patience = 3,
)

# Count the model parameters — record this in your thesis
total_params = sum(p.numel() for p in tft.parameters())
trainable    = sum(p.numel() for p in tft.parameters() if p.requires_grad)
print(f"\nModel built successfully")
print(f"  Total parameters    : {total_params:,}")
print(f"  Trainable parameters: {trainable:,}")
print(f"  Architecture        : hidden={HIDDEN_SIZE}, heads={ATTENTION_HEAD_SIZE}, "
      f"dropout={DROPOUT}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — TRAINING CALLBACKS
#
# WHAT ARE CALLBACKS?
#   Callbacks are functions that pytorch_lightning calls automatically
#   at specific points during training (end of each epoch, end of training, etc.)
#   You define what should happen, lightning handles when to call it.
#
# WE USE 3 CALLBACKS:
#
# 1. EarlyStopping:
#    Monitors validation loss after each epoch.
#    If val_loss doesn't improve for `patience=5` consecutive epochs,
#    training stops automatically and restores the best weights.
#
#    WHY IS THIS IMPORTANT?
#    Without early stopping, the model might keep training after it has
#    already found the best weights, causing it to overfit (memorize
#    training data instead of learning generalizable patterns).
#
#    EXAMPLE:
#    Epoch 10: val_loss = 0.655 (new best)
#    Epoch 11: val_loss = 0.658 (worse)
#    Epoch 12: val_loss = 0.661 (worse)
#    Epoch 13: val_loss = 0.659 (worse)
#    Epoch 14: val_loss = 0.663 (worse)
#    Epoch 15: val_loss = 0.660 (worse — 5 epochs without improvement)
#    → Training stops, best model from Epoch 10 is restored.
#
# 2. ModelCheckpoint:
#    Saves the model weights to disk after every epoch.
#    Only keeps the best model (lowest val_loss).
#    EXAMPLE filename: tft_bvmt_epoch=10_val_loss=0.6550.ckpt
#    This .ckpt file is what you download from Lightning AI.
#
# 3. LearningRateMonitor:
#    Logs the learning rate at each epoch so you can see when
#    ReduceLROnPlateau kicked in and halved the learning rate.
#    Useful for diagnosing training problems.
# ══════════════════════════════════════════════════════════════════════════

timestamp   = datetime.now().strftime('%Y%m%d_%H%M')
run_name    = f"{'smoke' if SMOKE_TEST else 'full'}_{timestamp}"

early_stop_callback = EarlyStopping(
    monitor   = 'val_loss',   # watch validation loss
    patience  = 5,             # stop after 5 epochs without improvement
    mode      = 'min',         # lower val_loss is better
    verbose   = True,          # print message when stopping
)

checkpoint_callback = ModelCheckpoint(
    dirpath   = str(MODEL_DIR),
    filename  = f'tft_bvmt_{run_name}' + '_{epoch:02d}_{val_loss:.4f}',
    monitor   = 'val_loss',
    mode      = 'min',
    save_top_k = 1,            # keep only the best model
    verbose   = True,
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

# CSV logger — saves training metrics to a readable file
csv_logger = CSVLogger(
    save_dir = str(LOG_DIR),
    name     = run_name,
)

print(f"\nCallbacks configured")
print(f"  Early stopping   : patience=5 epochs")
print(f"  Model checkpoint : saves to {MODEL_DIR}")
print(f"  Run name         : {run_name}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 — TRAINER AND TRAINING
#
# WHAT IS THE TRAINER?
#   The pytorch_lightning Trainer handles the entire training loop.
#   Without it you would have to write:
#     for epoch in range(MAX_EPOCHS):
#         for batch in train_dataloader:
#             output = model(batch)
#             loss = criterion(output, batch.target)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         validate()
#         save_checkpoint()
#         check_early_stopping()
#   The Trainer does all of this for you.
#
# KEY PARAMETERS:
#
# gradient_clip_val = 0.1:
#   Clips gradients to prevent "exploding gradients".
#   EXPLODING GRADIENTS EXPLANATION:
#   During backpropagation, gradients can grow exponentially large.
#   If unchecked, weight updates become enormous and the model
#   "explodes" — loss becomes NaN and training crashes.
#   gradient_clip_val=0.1 limits the maximum gradient norm,
#   keeping training stable. This is standard for transformers.
#
# accelerator + devices:
#   'gpu' + 1 = use 1 GPU (RTX 3050 or T4)
#   'cpu' + 0 = use CPU (slow, for testing only)
#
# precision = '16-mixed':
#   Uses 16-bit floating point (half precision) for most operations.
#   This halves GPU memory usage and speeds up training by ~2x.
#   pytorch_lightning automatically handles the precision conversion.
#   'full' precision (32-bit) would be safer but uses 2x more VRAM.
#   We use 'bf16-mixed' if Ampere GPU (RTX 3050) or '16-mixed' otherwise.
# ══════════════════════════════════════════════════════════════════════════

# Choose precision based on GPU capability
if torch.cuda.is_available():
    # Ampere architecture (RTX 30xx, RTX 40xx) supports bf16
    # Older architectures use fp16
    gpu_name = torch.cuda.get_device_name(0).lower()
    if any(x in gpu_name for x in ['3050', '3060', '3070', '3080', '3090',
                                    '4060', '4070', '4080', '4090', 't4', 'a100']):
        precision = 'bf16-mixed'
    else:
        precision = '16-mixed'
    print(f"Using mixed precision: {precision}")
else:
    precision = '32'

# Lightning rejects fractional limit_val_batches if it resolves to < 1 batch.
# In smoke mode, small datasets may produce a single validation batch.
if SMOKE_TEST and len(val_dataloader) < 2:
    limit_val_batches = 1
else:
    limit_val_batches = 0.5 if SMOKE_TEST else 1.0

trainer = Trainer(
    max_epochs         = MAX_EPOCHS,
    accelerator        = GPU_DEVICE,
    devices            = GPU_COUNT if GPU_COUNT > 0 else 'auto',
    gradient_clip_val  = 0.1,         # prevent exploding gradients
    precision          = precision,   # half precision for speed/memory
    callbacks          = cast(Any, [
        early_stop_callback,
        checkpoint_callback,
        lr_monitor,
    ]),
    logger             = cast(Any, csv_logger),
    enable_progress_bar = True,        # show tqdm progress bars
    log_every_n_steps  = 10,
    # Limit validation to save time during smoke test (safe for tiny val sets)
    limit_val_batches  = limit_val_batches,
)

print(f"\nTrainer configured")
print(f"  Max epochs   : {MAX_EPOCHS}")
print(f"  Accelerator  : {GPU_DEVICE}")
print(f"  Precision    : {precision}")
print(f"  Batch size   : {BATCH_SIZE}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 10 — START TRAINING
#
# WHAT HAPPENS DURING TRAINING:
#
# For each epoch:
#   1. For each batch of training samples:
#      a. FORWARD PASS: TFT reads the 60-day encoder input for each sample
#         in the batch and produces a probability for UP direction
#      b. LOSS COMPUTATION: CrossEntropy compares the prediction to
#         the actual direction_7d label
#      c. BACKWARD PASS: PyTorch computes gradients — how much each
#         weight contributed to the error
#      d. WEIGHT UPDATE: The Adam optimizer adjusts all weights slightly
#         in the direction that reduces the error
#   2. After all training batches: validation loop runs
#      → computes val_loss on 2024 data (model doesn't learn from this)
#   3. Callbacks run: checkpoint saved if val_loss improved,
#      early stopping check, learning rate check
#
# READING THE TRAINING OUTPUT:
#
# Epoch 1/30: 100%|████| 500/500 [02:15<00:00]  loss=0.6920  val_loss=0.6850
#              ^progress    ^batches              ^train loss  ^val loss
#
# What the loss numbers mean:
#   loss = 0.693: the random baseline (log(2) for binary classification)
#                 This means the model is guessing randomly
#   loss = 0.680: model is learning — it's slightly better than random
#   loss = 0.650: good learning — meaningful signal found
#   loss = 0.500: very strong — model has found reliable patterns
#                 WARNING: if this happens in epoch 1, check for data leakage!
#
# What to watch for:
#   GOOD: val_loss decreasing gradually each epoch
#   NEUTRAL: val_loss oscillating ±0.005 (normal noise)
#   BAD: val_loss increasing after epoch 5 (overfitting — dropout too low)
#   BAD: val_loss stuck at 0.693 after epoch 5 (not learning — check features)
#   STOP: val_loss = NaN (gradient explosion — reduce learning rate)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print()
print("Reading the training output:")
print("  loss     = training loss (how wrong on training data)")
print("  val_loss = validation loss (how wrong on 2024 data)")
print("  Both should DECREASE over epochs")
print("  val_loss < 0.693 = model is learning")
print("  val_loss < 0.670 = good progress")
print("  val_loss < 0.650 = strong result")
print()
print("Training started at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print()

# This is the one line that starts training
# Everything above was preparation — this is where the GPU starts working
trainer.fit(
    model             = cast(Any, tft),
    train_dataloaders = train_dataloader,
    val_dataloaders   = val_dataloader,
)

print()
print("Training finished at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# ══════════════════════════════════════════════════════════════════════════
# SECTION 11 — POST-TRAINING SUMMARY
#
# WHAT THIS DOES:
#   After training completes, prints a summary of results and
#   saves training metadata to a JSON file for your thesis records.
# ══════════════════════════════════════════════════════════════════════════

best_model_path = checkpoint_callback.best_model_path
best_val_loss   = checkpoint_callback.best_model_score

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Best val_loss  : {best_val_loss:.4f}")
print(f"  Best model     : {best_model_path}")
print(f"  Training logs  : {LOG_DIR / run_name}")
print()

# Interpret the final val_loss for the thesis
if best_val_loss is not None:
    val = float(best_val_loss)
    if val < 0.640:
        interpretation = "EXCELLENT — strong directional signal found in BVMT data"
    elif val < 0.660:
        interpretation = "GOOD — meaningful patterns learned"
    elif val < 0.680:
        interpretation = "ACCEPTABLE — slight improvement over random baseline"
    else:
        interpretation = "WEAK — near-random. Check features and class balance"
    print(f"  Interpretation : {interpretation}")

print()

# Save training metadata for thesis records
metadata = {
    'run_name'          : run_name,
    'smoke_test'        : SMOKE_TEST,
    'best_val_loss'     : float(best_val_loss) if best_val_loss else None,
    'best_model_path'   : str(best_model_path),
    'epochs_trained'    : trainer.current_epoch,
    'total_params'      : total_params,
    'hidden_size'       : HIDDEN_SIZE,
    'attention_heads'   : ATTENTION_HEAD_SIZE,
    'batch_size'        : BATCH_SIZE,
    'learning_rate'     : LEARNING_RATE,
    'encoder_length'    : MAX_ENCODER_LENGTH,
    'prediction_length' : MAX_PREDICTION_LENGTH,
    'train_rows'        : len(train_df),
    'val_rows'          : len(val_df[val_df['date'] >= TRAIN_END]),
    'class_balance_up'  : float(up_pct),
    'class_weighted'    : USE_CLASS_WEIGHTS,
    'stocks_used'       : int(pd.Series(df['ticker_id']).nunique()),
    'gpu_device'        : GPU_DEVICE,
    'precision'         : precision,
}

meta_path = RESULTS_DIR / f'training_metadata_{run_name}.json'
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  Metadata saved : {meta_path}")
print()
print("NEXT STEP:")
print("  Run: python training/evaluate_tft.py")
print(f"  Use model: {best_model_path}")
print("=" * 60)