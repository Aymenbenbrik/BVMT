"""training/evaluate_tft_v3_per_row.py

Per-row evaluation of the TFT v3 quantile model on the 2025 test set,
with persistence of every prediction (Q10, Q50, Q90, ticker, date,
true 7-day return) to a CSV that downstream scripts can aggregate.

The existing training/evaluate_tft.py targets the v1 binary-classifier
checkpoint and reports a single aggregate accuracy. v3 (quantile) does
not have a reproducible test-time evaluator yet, so the headline 77.4%
in the article is not regenerable from the public artefacts. This
script closes that gap. Produces the input that
scripts/per_stock_breakdown.py needs for D3 of
paper/critique_independent.md.

USAGE
    # default: load the most recent v3 checkpoint, write CSV to results/
    python training/evaluate_tft_v3_per_row.py

    # explicit checkpoint and output path
    python training/evaluate_tft_v3_per_row.py \\
        --ckpt models/tft_quantile_v2_fixed_*.ckpt \\
        --out  results/tft_v3_predictions_2025.csv

REQUIREMENTS
    - models/tft_quantile_*.ckpt  (the v3 checkpoint)
    - data/features/tft_features.csv (post-cleaning feature file)
    - pytorch, pytorch_lightning, pytorch_forecasting

OUTPUT CSV columns
    ticker, ticker_id, isin_code, date, time_idx,
    q10, q50, q90,
    true_return_7d, true_direction,
    pred_direction_R1   (UP iff q50 >= 0; the rule documented in
                         Section V.B of the article)

Decoupling rationale: this script writes RAW predictions; the
per-stock aggregation lives in scripts/per_stock_breakdown.py so the
same predictions can also drive the rule-sensitivity sweep of
Action 1.5b without re-running the model.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

try:
    from lightning.pytorch import Trainer
except ImportError:
    from pytorch_lightning import Trainer


DATA_PATH = ROOT / "data" / "features" / "tft_features.csv"
DEFAULT_CKPT_GLOB = "tft_quantile_*.ckpt"
DEFAULT_OUT = ROOT / "results" / "tft_v3_predictions_2025.csv"

# Must match training/train_tft_v3.py exactly
TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"
MAX_ENCODER_LENGTH = 30
MAX_PREDICTION_LENGTH = 7
RETURN_CLIP_MIN = -0.15
RETURN_CLIP_MAX = +0.15

TIME_VARYING_FEATURES = [
    "close_price", "open_price", "high_price", "low_price",
    "volume", "daily_return", "ma_5", "ma_20", "ma_50",
    "rsi_14", "volatility_20", "volume_ma_20",
    "price_to_ma20_ratio", "rsi_momentum", "volume_spike",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=None,
                   help=f"Path to a v3 checkpoint. Defaults to most recent models/{DEFAULT_CKPT_GLOB}.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output CSV path.")
    p.add_argument("--data", type=Path, default=DATA_PATH,
                   help="Input feature file (must match the training input).")
    p.add_argument("--limit-batches", type=int, default=None,
                   help="Stop after N batches (for smoke testing the pipeline).")
    return p.parse_args()


def find_default_ckpt() -> Path:
    candidates = sorted((ROOT / "models").glob(DEFAULT_CKPT_GLOB),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(
            f"No v3 checkpoint found under models/ matching {DEFAULT_CKPT_GLOB}. "
            "Train one with `python training/train_tft_v3.py` or pass --ckpt."
        )
    return candidates[0]


def build_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate future_return_7d and _target_date as in train_tft_v3 Sec. 4."""
    df = df.sort_values(["ticker_id", "time_idx"]).reset_index(drop=True)
    df["_future_close"] = (
        df.groupby("ticker_id")["close_price"]
          .transform(lambda x: x.shift(-MAX_PREDICTION_LENGTH))
    )
    df["_target_date"] = (
        df.groupby("ticker_id")["date"]
          .transform(lambda s: s.shift(-MAX_PREDICTION_LENGTH))
    )
    df["future_return_7d"] = (
        (df["_future_close"] - df["close_price"])
        / df["close_price"].replace(0, np.nan)
    ).fillna(np.nan)
    df = df.dropna(subset=["future_return_7d", "_target_date"]).reset_index(drop=True)
    df = df.drop(columns=["_future_close"])
    df["future_return_7d"] = df["future_return_7d"].clip(RETURN_CLIP_MIN, RETURN_CLIP_MAX).astype("float32")
    df["true_direction"] = (df["future_return_7d"] >= 0).astype(int)
    return df


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    if "price_to_ma20_ratio" not in df.columns:
        df["price_to_ma20_ratio"] = (
            df["close_price"] / df["ma_20"].replace(0, np.nan)
        ).fillna(1.0).clip(0.5, 2.0).astype("float32")
    if "rsi_momentum" not in df.columns:
        df["rsi_momentum"] = (
            df.groupby("ticker_id")["rsi_14"]
              .transform(lambda x: x - x.shift(5))
              .fillna(0.0)
              .astype("float32")
        )
    if "volume_spike" not in df.columns:
        df["volume_spike"] = (
            df["volume"] / df["volume_ma_20"].replace(0, np.nan)
        ).fillna(1.0).clip(0.0, 10.0).astype("float32")
    return df


def main() -> int:
    args = parse_args()
    ckpt = args.ckpt or find_default_ckpt()
    print(f"Checkpoint     : {ckpt}")
    print(f"Data           : {args.data}")
    print(f"Output CSV     : {args.out}")

    print("\nLoading data...")
    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker_id"] = df["ticker_id"].astype(str)
    df["company_type_id"] = df["company_type_id"].astype(str)
    df["time_idx"] = df["time_idx"].astype("int32")
    for col in df.columns:
        if col in {"ticker", "ticker_id", "company_type_id", "date", "isin_code", "direction_7d", "time_idx"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")

    df = build_target_columns(df)
    df = add_relative_features(df)

    test_end_anchor = pd.to_datetime(VAL_END)
    full_df = df[df["_target_date"] >= test_end_anchor].copy()
    if full_df.empty:
        raise SystemExit("Test split is empty -- check VAL_END and the data file.")

    # Training-equivalent dataset (encoder context = full history, only test
    # rows generate predictions). We rebuild the dataset as in train_tft_v3.
    train_clean = df.copy()
    train_clean["future_return_7d"] = train_clean["future_return_7d"].astype("float32")

    print(f"\nBuilding TimeSeriesDataSet (encoder context = full history)...")
    ds_full = TimeSeriesDataSet(
        cast(pd.DataFrame, train_clean),
        time_idx="time_idx",
        group_ids=["ticker_id"],
        target="future_return_7d",
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["ticker_id", "company_type_id"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=TIME_VARYING_FEATURES,
        target_normalizer=GroupNormalizer(groups=["ticker_id"], transformation=None),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # Test subset: rows whose target date is in the 2025 test window.
    test_only = train_clean[train_clean["_target_date"] >= test_end_anchor].reset_index(drop=True)
    test_dataset = TimeSeriesDataSet.from_dataset(
        ds_full, cast(pd.DataFrame, test_only), predict=True, stop_randomization=True
    )
    test_loader = test_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    print(f"  Test rows : {len(test_only):,}")

    print("\nLoading model from checkpoint...")
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"  Device : {device}")

    print("\nRunning inference...")
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, logger=False, enable_progress_bar=True,
    )
    raw_preds = trainer.predict(model, test_loader)

    # Collect Q10/Q50/Q90 + decoder time_idx ranges
    rows = []
    for batch_idx, batch_output in enumerate(raw_preds or []):
        if args.limit_batches is not None and batch_idx >= args.limit_batches:
            break
        if isinstance(batch_output, (list, tuple)):
            preds = batch_output[0]
            x = batch_output[1] if len(batch_output) > 1 else None
        else:
            preds = batch_output
            x = None
        if preds.dim() == 4:
            preds = preds[:, 0, :, :]  # take horizon-step 0
        # preds shape: [batch, n_quantiles] for the chosen horizon step
        preds_np = preds.detach().cpu().numpy()
        for i in range(preds_np.shape[0]):
            row = {
                "q10": float(preds_np[i, 0]),
                "q50": float(preds_np[i, 1]) if preds_np.shape[1] >= 2 else float(preds_np[i, 0]),
                "q90": float(preds_np[i, 2]) if preds_np.shape[1] >= 3 else float(preds_np[i, 0]),
            }
            rows.append(row)

    if not rows:
        raise SystemExit("No predictions produced; aborting.")

    pred_df = pd.DataFrame(rows)
    test_only_aligned = test_only.iloc[: len(pred_df)].reset_index(drop=True)
    out = pd.DataFrame({
        "ticker":           test_only_aligned["ticker"].values,
        "ticker_id":        test_only_aligned["ticker_id"].values,
        "isin_code":        test_only_aligned.get("isin_code", pd.Series([""] * len(pred_df))).values,
        "date":             pd.to_datetime(test_only_aligned["date"]).dt.strftime("%Y-%m-%d").values,
        "time_idx":         test_only_aligned["time_idx"].values,
        "q10":              pred_df["q10"].values,
        "q50":              pred_df["q50"].values,
        "q90":              pred_df["q90"].values,
        "true_return_7d":   test_only_aligned["future_return_7d"].values,
        "true_direction":   test_only_aligned["true_direction"].values,
    })
    out["pred_direction_R1"] = (out["q50"] >= 0).astype(int)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    overall_acc = (out["pred_direction_R1"] == out["true_direction"]).mean() * 100
    print(f"\nOverall test accuracy (R1 = sign of Q50): {overall_acc:.2f}%")
    print(f"Wrote {len(out):,} predictions to: {args.out}")
    print("\nNext step: python scripts/per_stock_breakdown.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
