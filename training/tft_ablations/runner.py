"""training/tft_ablations/runner.py

Single entry point for the three TRAINING-time TFT ablations of
Action 1.4. Mirrors the data preparation pipeline of
training/train_tft_v3.py (anti-leak Section 5b included) but lives in
its own module so the published v3 reference script stays untouched.

USAGE
    # Smoke test (3 stocks, 5 epochs, ~3 min CPU). Confirms each variant
    # instantiates and trains a couple of mini-batches without crashing.
    python -m training.tft_ablations.runner --variant baseline    --smoke
    python -m training.tft_ablations.runner --variant no_vsn      --smoke
    python -m training.tft_ablations.runner --variant global_norm --smoke

    # Vertex AI Custom Job (full T4/A100 GPU run, ~6-12 h). The Vertex
    # entrypoint in vertex/submit.py forwards --variant and --epochs.
    python -m training.tft_ablations.runner --variant no_vsn --epochs 100

OUTPUTS (per run)
    models/tft_ablations/<variant>_<timestamp>.ckpt
    results/tft_ablations/<variant>_summary.json
        (best_val_loss, best_q50_dir_acc, params, normalizer class,
         vsn_patched count, epochs trained, run_name)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet  # noqa: E402
from pytorch_forecasting.metrics import QuantileLoss  # noqa: E402

try:
    from lightning.pytorch import Trainer  # type: ignore
    from lightning.pytorch.callbacks import (  # type: ignore
        Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint,
    )
    from lightning.pytorch.loggers import CSVLogger  # type: ignore
except ImportError:
    from pytorch_lightning import Trainer  # type: ignore
    from pytorch_lightning.callbacks import (  # type: ignore
        Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint,
    )
    from pytorch_lightning.loggers import CSVLogger  # type: ignore

from training.tft_ablations.variants import (  # noqa: E402
    apply_no_vsn_patch, make_target_normalizer,
)


# -----------------------------------------------------------------------------
# Constants — must match training/train_tft_v3.py exactly so the ablations
# are diff-only versus the baseline (any divergence is the ablation, not the
# data path).
# -----------------------------------------------------------------------------

TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"
QUANTILES = [0.1, 0.5, 0.9]
RETURN_CLIP_MIN = -0.15
RETURN_CLIP_MAX = +0.15

TIME_VARYING_FEATURES = [
    "close_price", "open_price", "high_price", "low_price",
    "volume", "daily_return", "ma_5", "ma_20", "ma_50",
    "rsi_14", "volatility_20", "volume_ma_20",
    "price_to_ma20_ratio", "rsi_momentum", "volume_spike",
]

NON_NUMERIC_COLS = {
    "ticker", "ticker_id", "company_type_id",
    "date", "direction_7d", "time_idx", "isin_code",
}


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed_from_env(default: int = 0) -> int:
    seed_env = os.environ.get("BVMT_SEED")
    seed = int(seed_env) if seed_env is not None else default
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


# -----------------------------------------------------------------------------
# Data loading + anti-leak (Sections 3-6 of train_tft_v3.py)
# -----------------------------------------------------------------------------

def _add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
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
    return df


def load_and_prepare(data_path: Path, smoke_stocks: int = 0,
                     prediction_length: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    for col in df.columns:
        if col in NON_NUMERIC_COLS:
            continue
        try:
            if str(df[col].dtype) == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")
            else:
                df[col] = df[col].astype("float32")
        except (ValueError, TypeError):
            pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    nan_count = int(df[numeric_cols].isna().sum().sum())
    if nan_count > 0:
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

    df["time_idx"] = df["time_idx"].astype("int32")
    df["ticker_id"] = df["ticker_id"].astype(str)
    df["company_type_id"] = df["company_type_id"].astype(str)
    df = df.sort_values(["ticker_id", "time_idx"]).reset_index(drop=True)

    # Section 4: future_return_7d + _target_date in one pass
    df["_future_close"] = df.groupby("ticker_id")["close_price"].transform(
        lambda x: x.shift(-prediction_length))
    df["_target_date"] = df.groupby("ticker_id")["date"].transform(
        lambda s: s.shift(-prediction_length))
    df["future_return_7d"] = (
        (df["_future_close"] - df["close_price"]) / df["close_price"].replace(0, np.nan)
    )
    df = df.dropna(subset=["future_return_7d", "_target_date"]).reset_index(drop=True)
    df["future_return_7d"] = df["future_return_7d"].clip(
        RETURN_CLIP_MIN, RETURN_CLIP_MAX).astype("float32")
    df = df.drop(columns=["_future_close"])

    df = _add_relative_features(df)

    # Section 5b: anti-leak boundary drop
    train_hi = pd.to_datetime(TRAIN_END)
    val_hi = pd.to_datetime(VAL_END)
    mask_train_leak = (df["date"] < train_hi) & (df["_target_date"] >= train_hi)
    mask_val_leak = (
        (df["date"] >= train_hi)
        & (df["date"] < val_hi)
        & (df["_target_date"] >= val_hi)
    )
    n_leak = int((mask_train_leak | mask_val_leak).sum())
    df = df[~(mask_train_leak | mask_val_leak)].reset_index(drop=True)
    df = df.drop(columns=["_target_date"])

    # Smoke filter
    if smoke_stocks > 0:
        keep = sorted(df["ticker_id"].unique().tolist())[:smoke_stocks]
        df = df[df["ticker_id"].isin(keep)].copy().reset_index(drop=True)

    train_df = df[df["date"] < TRAIN_END].copy().reset_index(drop=True)
    val_df = df[df["date"] < VAL_END].copy().reset_index(drop=True)

    print(f"  Anti-leak dropped : {n_leak:,} rows")
    print(f"  Train rows        : {len(train_df):,}")
    print(f"  Val   rows        : {len(val_df):,}  "
          f"({(val_df['date'] >= TRAIN_END).sum():,} are 2024)")
    return train_df, val_df


def build_datasets(variant: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   max_encoder_length: int, max_prediction_length: int):
    target_normalizer = make_target_normalizer(variant)

    training_dataset = TimeSeriesDataSet(
        cast(pd.DataFrame, train_df.reset_index(drop=True)),
        time_idx="time_idx",
        group_ids=["ticker_id"],
        target="future_return_7d",
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["ticker_id", "company_type_id"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=TIME_VARYING_FEATURES,
        target_normalizer=target_normalizer,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    val_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        cast(pd.DataFrame, val_df.reset_index(drop=True)),
        predict=True,
        stop_randomization=True,
    )
    return training_dataset, val_dataset


# -----------------------------------------------------------------------------
# Lightweight Q50 directional accuracy callback
# -----------------------------------------------------------------------------

class _DirAcc(Callback):
    def __init__(self, val_loader, q50_idx: int = 1, every_n: int = 3):
        super().__init__()
        self.val_loader = val_loader
        self.q50_idx = q50_idx
        self.every_n = every_n
        self.history: list[dict] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n != 0:
            return
        try:
            pl_module.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in self.val_loader:
                    x = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                         for k, v in x.items()}
                    out = pl_module(x)
                    pred = out.prediction[:, -1, self.q50_idx]
                    actual = (y[0][:, -1] if y[0].dim() > 1 else y[0]).to(pl_module.device)
                    correct += ((pred > 0) == (actual > 0)).sum().item()
                    total += len(actual)
                    if total >= 3000:
                        break
            if total:
                acc = correct / total * 100
                self.history.append({"epoch": epoch, "dir_acc": round(acc, 2)})
                print(f"  [Q50 dir_acc @ epoch {epoch}] {acc:.1f}%")
        except Exception as exc:
            print(f"  [DirAcc skipped: {exc}]")


# -----------------------------------------------------------------------------
# Train one variant
# -----------------------------------------------------------------------------

def build_and_train(variant: str, training_dataset, val_dataset,
                    *, max_epochs: int, batch_size: int,
                    hidden_size: int, attention_head_size: int,
                    dropout: float, hidden_continuous_size: int,
                    learning_rate: float, weight_decay: float,
                    out_dir: Path, run_name: str, smoke: bool):
    train_loader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0)
    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=len(QUANTILES),
        loss=QuantileLoss(quantiles=QUANTILES),
        weight_decay=weight_decay,
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=7,
    )

    n_vsn_patched = 0
    vsn_patched_names: list[str] = []
    if variant == "no_vsn":
        n_vsn_patched, vsn_patched_names = apply_no_vsn_patch(tft)
        print(f"  Patched {n_vsn_patched} VSN module(s): {vsn_patched_names}")
        if n_vsn_patched != 3:
            print(f"  WARNING: expected 3 VSN modules, found {n_vsn_patched}")

    n_params = sum(p.numel() for p in tft.parameters())
    print(f"  Params : {n_params:,}")

    if torch.cuda.is_available():
        accelerator, devices = "gpu", 1
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    else:
        accelerator, devices = "cpu", "auto"
        precision = "32"
    print(f"  Device : {accelerator} ({precision})")

    ckpt_dir = out_dir / "models"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    early_stop = EarlyStopping(monitor="val_loss", patience=15, mode="min", verbose=False)
    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir), filename=f"{run_name}" + "_{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=1, verbose=False,
    )
    dir_acc_cb = _DirAcc(val_loader)
    lim_val = 1 if (smoke and len(val_loader) < 2) else (0.5 if smoke else 1.0)
    # Smoke aggressively caps train batches so the run completes in
    # ~30-60 s on CPU. The point of the smoke is "does it crash / does
    # the VSN patch + the global normalizer instantiate", not "does it
    # learn". The full sweep on Vertex AI removes both limits.
    lim_train = 8 if smoke else 1.0

    callbacks_smoke = [early_stop, checkpoint,
                       LearningRateMonitor(logging_interval="epoch")]
    callbacks_full = callbacks_smoke + [dir_acc_cb]

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator, devices=devices,
        gradient_clip_val=0.3, precision=precision,
        callbacks=cast(Any, callbacks_smoke if smoke else callbacks_full),
        logger=cast(Any, CSVLogger(save_dir=str(log_dir), name=run_name)),
        enable_progress_bar=smoke, log_every_n_steps=10,
        limit_train_batches=lim_train,
        limit_val_batches=lim_val,
    )

    t0 = time.time()
    trainer.fit(model=cast(Any, tft),
                train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.time() - t0

    best_val_loss = (float(checkpoint.best_model_score)
                     if checkpoint.best_model_score is not None else None)
    best_path = checkpoint.best_model_path

    return {
        "variant": variant,
        "run_name": run_name,
        "best_val_loss": best_val_loss,
        "best_ckpt": best_path,
        "epochs_trained": trainer.current_epoch,
        "params": n_params,
        "vsn_patched": n_vsn_patched,
        "vsn_patched_names": vsn_patched_names,
        "target_normalizer": type(training_dataset.target_normalizer).__name__,
        "dir_acc_history": dir_acc_cb.history,
        "best_q50_dir_acc": (
            max((h["dir_acc"] for h in dir_acc_cb.history), default=None)
        ),
        "elapsed_seconds": round(elapsed, 1),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["baseline", "no_vsn", "global_norm"],
                   required=True)
    p.add_argument("--data", type=Path,
                   default=ROOT / "data" / "features" / "tft_features.csv")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "tft_ablations")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: 3 stocks, 5 epochs, tiny model.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden-continuous-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--encoder-length", type=int, default=30)
    p.add_argument("--prediction-length", type=int, default=7)
    p.add_argument("--gcs-output", type=str, default="",
                   help="Optional GCS prefix (gs://bucket/path) to upload "
                        "the contents of --out to after training. Used by "
                        "Vertex AI Custom Jobs since the container is "
                        "ephemeral. AIP_MODEL_DIR env var is used as a "
                        "fallback if this flag is empty.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seed = set_seed_from_env(default=0)
    print(f"BVMT_SEED -> {seed}")
    print(f"Variant   : {args.variant}{'  [smoke]' if args.smoke else ''}")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        smoke_stocks = 3
        max_epochs = min(args.epochs, 5)
        hidden_size = 16
        attention_heads = 2
        hidden_continuous_size = 8
        batch_size = 16
    else:
        smoke_stocks = 0
        max_epochs = args.epochs
        hidden_size = args.hidden_size
        attention_heads = args.attention_heads
        hidden_continuous_size = args.hidden_continuous_size
        batch_size = args.batch_size

    print(f"\nLoading {args.data}...")
    train_df, val_df = load_and_prepare(
        args.data, smoke_stocks=smoke_stocks,
        prediction_length=args.prediction_length,
    )

    print(f"\nBuilding TimeSeriesDataSet (target normalizer for variant={args.variant})...")
    training_dataset, val_dataset = build_datasets(
        args.variant, train_df, val_df,
        max_encoder_length=args.encoder_length,
        max_prediction_length=args.prediction_length,
    )
    print(f"  Train samples : {len(training_dataset):,}")
    print(f"  Val   samples : {len(val_dataset):,}")
    print(f"  Normalizer    : {type(training_dataset.target_normalizer).__name__}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"tft_{args.variant}_{'smoke' if args.smoke else 'full'}_{timestamp}"

    summary = build_and_train(
        args.variant, training_dataset, val_dataset,
        max_epochs=max_epochs, batch_size=batch_size,
        hidden_size=hidden_size, attention_head_size=attention_heads,
        dropout=args.dropout, hidden_continuous_size=hidden_continuous_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        out_dir=args.out, run_name=run_name, smoke=args.smoke,
    )
    summary["seed"] = seed
    summary["smoke"] = args.smoke

    print("\n=== Summary ===")
    print(f"  variant           = {summary['variant']}")
    print(f"  best_val_loss     = {summary['best_val_loss']}")
    print(f"  best_q50_dir_acc  = {summary['best_q50_dir_acc']}")
    print(f"  target_normalizer = {summary['target_normalizer']}")
    print(f"  vsn_patched       = {summary['vsn_patched']}")
    print(f"  epochs_trained    = {summary['epochs_trained']}")
    print(f"  best_ckpt         = {summary['best_ckpt']}")
    print(f"  elapsed           = {summary['elapsed_seconds']}s")

    out_json = args.out / f"{args.variant}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating,))
                  else int(o) if isinstance(o, (np.integer,)) else str(o))
    print(f"  Wrote: {out_json}")

    if summary["best_ckpt"]:
        stable_path = args.out / "models" / f"{args.variant}_best.ckpt"
        try:
            shutil.copy(summary["best_ckpt"], stable_path)
            print(f"  Copied to stable path: {stable_path}")
        except Exception as exc:
            print(f"  Stable copy skipped: {exc}")

    # Vertex AI: upload everything to a GCS path. Either explicit via
    # --gcs-output, or via the AIP_MODEL_DIR env var (set by Vertex when
    # baseOutputDirectory is configured). The training container is
    # ephemeral, so without this step the checkpoints + summary JSON
    # would be lost when the job finishes. No-op for local runs (both
    # the flag and the env var unset / empty).
    aip_dir = (args.gcs_output or os.environ.get("AIP_MODEL_DIR", "")).strip()
    if aip_dir.startswith("gs://"):
        try:
            from google.cloud import storage  # type: ignore
            rest = aip_dir[5:]
            bucket_name, _, prefix = rest.partition("/")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            n_uploaded = 0
            for f in args.out.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(args.out).as_posix()
                blob_name = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
                bucket.blob(blob_name).upload_from_filename(str(f))
                n_uploaded += 1
            print(f"  Uploaded {n_uploaded} file(s) to {aip_dir}")
        except Exception as exc:
            print(f"  GCS upload failed (non-fatal): {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
