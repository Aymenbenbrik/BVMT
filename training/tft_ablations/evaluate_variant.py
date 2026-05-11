"""training/tft_ablations/evaluate_variant.py

Per-variant test-set evaluation for the Action 1.4 ablation
checkpoints. Mirrors training/evaluate_tft_v3_per_row.py but with two
variant-specific corrections that the generic eval script does not
perform:

  - target_normalizer matches what the variant was TRAINED with
    (GroupNormalizer for baseline/no_vsn, TorchNormalizer for
    global_norm). Otherwise the test dataset normalises targets
    differently from what the model expects, and the inverse
    transform on the model's quantile output gives wrong-sign
    predictions.
  - For variant=no_vsn, the same uniform-VSN monkey-patch that was
    applied at training time is reapplied to the loaded model.
    Without this, load_from_checkpoint restores the softmax-gated
    forward (only state_dict is serialised, not the forward
    override) and the trained weights are routed through a
    different gating function than they were trained against.

USAGE
    python -m training.tft_ablations.evaluate_variant \\
        --variant baseline \\
        --ckpt   results/tft_ablations/vertex/baseline_best.ckpt

OUTPUT
    results/tft_ablations/vertex/<variant>_test_predictions.csv
    Columns: ticker, ticker_id, date, q10, q50, q90,
             true_return_7d, true_direction, pred_direction_R1
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet  # noqa: E402

try:
    from lightning.pytorch import Trainer  # type: ignore
except ImportError:
    from pytorch_lightning import Trainer  # type: ignore

from training.tft_ablations.variants import (  # noqa: E402
    apply_no_vsn_patch, make_target_normalizer,
)


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
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--variant", required=True,
                   choices=["baseline", "no_vsn", "global_norm"])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path,
                   default=ROOT / "data" / "features" / "tft_features.csv")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV. Defaults to "
                        "results/tft_ablations/vertex/<variant>_test_predictions.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.out is None:
        args.out = (ROOT / "results" / "tft_ablations" / "vertex"
                    / f"{args.variant}_test_predictions.csv")
    print(f"Variant     : {args.variant}")
    print(f"Checkpoint  : {args.ckpt}")
    print(f"Output      : {args.out}")

    # ----- load + prepare data (same as train_tft_v3 / evaluate_tft_v3_per_row)
    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker_id"] = df["ticker_id"].astype(str)
    df["company_type_id"] = df["company_type_id"].astype(str)
    df["time_idx"] = df["time_idx"].astype("int32")
    for col in df.columns:
        if col in {"ticker", "ticker_id", "company_type_id", "date",
                   "isin_code", "direction_7d", "time_idx"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")

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
    df["future_return_7d"] = (df["future_return_7d"]
                               .clip(RETURN_CLIP_MIN, RETURN_CLIP_MAX).astype("float32"))
    df["true_direction"] = (df["future_return_7d"] >= 0).astype(int)

    if "price_to_ma20_ratio" not in df.columns:
        df["price_to_ma20_ratio"] = (
            df["close_price"] / df["ma_20"].replace(0, np.nan)
        ).fillna(1.0).clip(0.5, 2.0).astype("float32")
    if "rsi_momentum" not in df.columns:
        df["rsi_momentum"] = (
            df.groupby("ticker_id")["rsi_14"]
              .transform(lambda x: x - x.shift(5))
              .fillna(0.0).astype("float32")
        )
    if "volume_spike" not in df.columns:
        df["volume_spike"] = (
            df["volume"] / df["volume_ma_20"].replace(0, np.nan)
        ).fillna(1.0).clip(0.0, 10.0).astype("float32")

    test_end_anchor = pd.to_datetime(VAL_END)
    test_only = df[df["_target_date"] >= test_end_anchor].reset_index(drop=True)
    print(f"  Test rows : {len(test_only):,}")

    # ----- variant-aware dataset construction
    target_normalizer = make_target_normalizer(args.variant)
    print(f"  Normalizer: {type(target_normalizer).__name__}")

    ds_full = TimeSeriesDataSet(
        cast(pd.DataFrame, df.reset_index(drop=True)),
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
        target_normalizer=target_normalizer,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    # predict=False so the dataset emits every valid sliding-window
    # prediction in the test slice, not just the last window per ticker.
    # Matches the N_test=10,314 figure in the article (all daily 2025
    # prediction points across 43 active tickers, not 1-per-ticker).
    test_dataset = TimeSeriesDataSet.from_dataset(
        ds_full, cast(pd.DataFrame, test_only),
        predict=False, stop_randomization=True,
    )
    test_loader = test_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    # ----- load model and reapply variant-specific runtime patches
    print("\nLoading model from checkpoint...")
    model = TemporalFusionTransformer.load_from_checkpoint(str(args.ckpt))
    if args.variant == "no_vsn":
        n_patched, names = apply_no_vsn_patch(model)
        print(f"  Re-applied no_vsn patch to {n_patched} VSN module(s): {names}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"  Device    : {device}")

    # ----- inference. pytorch-forecasting 1.7 added a required
    # PredictCallback that Trainer.predict does not auto-inject; the
    # supported path is model.predict(loader, mode="quantiles").
    # return_index=True returns a Prediction tuple whose .index gives
    # the (ticker_id, time_idx) of each prediction, decoded from the
    # categorical encoder. We need this for predict=False because the
    # number of generated predictions (~10,924) does not match the
    # source DataFrame iloc-by-iloc.
    print("\nRunning inference on 2025 test set...")
    with torch.no_grad():
        preds = model.predict(test_loader, mode="quantiles",
                              return_index=True,
                              trainer_kwargs={"logger": False,
                                              "enable_progress_bar": False})

    pred_index = None
    if hasattr(preds, "index") and hasattr(preds, "output"):
        pred_tensor = preds.output
        pred_index = preds.index
    elif isinstance(preds, (list, tuple)) and len(preds) >= 2:
        pred_tensor = preds[0]
        pred_index = preds[1]
    else:
        pred_tensor = preds
    if isinstance(pred_tensor, (list, tuple)):
        pred_tensor = pred_tensor[0]
    pred_tensor = pred_tensor.detach().cpu() if isinstance(pred_tensor, torch.Tensor) else pred_tensor
    if pred_tensor.dim() == 4:
        # [batch, horizon, n_quantiles, ?] -> take horizon-step 0
        pred_tensor = pred_tensor[:, 0, :]
    if pred_tensor.dim() == 3:
        # [batch, horizon, n_quantiles] -> take LAST horizon step (the
        # 7-day-ahead point that future_return_7d targets)
        pred_tensor = pred_tensor[:, -1, :]
    preds_np = pred_tensor.numpy()

    rows = []
    for i in range(preds_np.shape[0]):
        rows.append({
            "q10": float(preds_np[i, 0]),
            "q50": float(preds_np[i, 1]) if preds_np.shape[1] >= 2 else float(preds_np[i, 0]),
            "q90": float(preds_np[i, 2]) if preds_np.shape[1] >= 3 else float(preds_np[i, 0]),
        })

    if not rows:
        raise SystemExit("No predictions produced; aborting.")
    pred_df = pd.DataFrame(rows)

    # Align predictions to source rows via the prediction-side index
    # returned by model.predict(..., return_index=True). It contains
    # (ticker_id, time_idx) for each prediction. The time_idx in this
    # frame is the FIRST decoder step (= row right after the encoder
    # window). The 7-day target was attached to the row whose
    # _target_date was 7 days later, but we computed direction from
    # future_return_7d which is anchored at the source row in
    # test_only at time_idx-MAX_PREDICTION_LENGTH+1. Practically the
    # cleanest merge key is (ticker_id, time_idx) because predict
    # returns the decoder-start time and we want the truth value at
    # that same row in test_only.
    if pred_index is None:
        raise SystemExit("model.predict did not return an index; "
                         "cannot align predictions back to source rows.")
    pi = pred_index.copy()
    pi["ticker_id"] = pi["ticker_id"].astype(str)
    pi["time_idx"] = pi["time_idx"].astype(int)
    merge_keys = ["ticker_id", "time_idx"]
    merged = pi[merge_keys].merge(
        test_only[["ticker", "ticker_id", "isin_code", "date",
                   "time_idx", "future_return_7d", "true_direction"]],
        on=merge_keys, how="left",
    )
    out = pd.DataFrame({
        "ticker":         merged["ticker"].values,
        "ticker_id":      merged["ticker_id"].values,
        "isin_code":      merged.get("isin_code", pd.Series([""] * len(merged))).values,
        "date":           pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d").values,
        "time_idx":       merged["time_idx"].values,
        "q10":            pred_df["q10"].values,
        "q50":            pred_df["q50"].values,
        "q90":            pred_df["q90"].values,
        "true_return_7d": merged["future_return_7d"].values,
        "true_direction": merged["true_direction"].values,
    })
    before = len(out)
    out = out.dropna(subset=["true_direction"]).reset_index(drop=True)
    if len(out) != before:
        print(f"  Dropped {before - len(out)} unmatched prediction rows "
              f"(no truth in test_only for that ticker_id+time_idx)")
    out["true_direction"] = out["true_direction"].astype(int)
    out["pred_direction_R1"] = (out["q50"] >= 0).astype(int)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    overall_acc = (out["pred_direction_R1"] == out["true_direction"]).mean() * 100
    print(f"\n[{args.variant}] Test accuracy (R1 = sign of Q50): {overall_acc:.2f}% on {len(out):,} rows")
    print(f"  Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
