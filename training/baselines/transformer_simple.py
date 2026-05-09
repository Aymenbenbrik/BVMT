"""training/baselines/transformer_simple.py

Action 1.3 of paper/improvement_plan.md, standard-Transformer half.
The architecture is the smallest the article's improvement plan
explicitly names: 2 transformer encoder blocks with d=64, 4 heads,
fed by the same 60-day x 15-feature window as CNN-LSTM and the
tabular ML baselines, with sinusoidal positional encoding.

The point is to isolate the Variable Selection Network's
contribution from raw multi-head self-attention. If a vanilla
Transformer also lands in the 50-53% band, the gap to TFT v3 is
mostly attributable to VSN + per-encoder-window normalisation +
quantile regression rather than to attention itself.

USAGE
    # Smoke (~1 min on CPU)
    python -m training.baselines.transformer_simple --smoke 3 --max-epochs 3

    # Full run (GPU recommended; on CPU expect 1-3 hours)
    python -m training.baselines.transformer_simple --max-epochs 15

OUTPUTS
    results/baselines/transformer_simple_predictions.csv
    results/baselines/transformer_simple_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from training.baselines._deep_common import (  # noqa: E402
    FEATURES,
    load_split,
    select_device,
    set_seed_from_env,
    train_classifier,
)
from training.baselines.metrics import evaluate_model  # noqa: E402


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class StandardTransformerClassifier(nn.Module):
    def __init__(
        self,
        n_features: int = len(FEATURES),
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.3,
        max_len: int = 200,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        h_last = h[:, -1, :]
        return self.fc(self.dropout(h_last))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "features" / "tft_features.csv")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "baselines")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=1000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seed = set_seed_from_env(default=0)
    print(f"BVMT_SEED -> {seed}")

    print(f"\nLoading {args.data} ...")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te, tk_te, dt_te) = load_split(args.data, args.window, args.smoke)
    if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
        raise SystemExit("Empty train or test split.")

    device = select_device()
    print(f"  Device: {device}")

    model = StandardTransformerClassifier(
        n_features=X_tr.shape[2], d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
        max_len=max(args.window + 4, 200),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nStandardTransformer: d_model={args.d_model}, h={args.n_heads}, layers={args.n_layers}, params={n_params:,}")

    res = train_classifier(
        model, X_tr, y_tr, X_va, y_va, X_te, y_te,
        device=device, max_epochs=args.max_epochs,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience,
        log_prefix="TRF ",
    )

    print(f"\nDone. best_val_acc={res.val_acc:.4f}  test_acc={res.test_acc:.4f}  ({res.elapsed_s:.0f}s)")

    summary = {
        "model": "transformer_simple",
        "hparams": {
            "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers,
            "dropout": args.dropout, "window": args.window, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay, "max_epochs": args.max_epochs,
            "patience": args.patience, "params": n_params, "seed": seed,
        },
        "val_acc": round(res.val_acc, 4),
        "best_epoch": res.best_epoch,
        "elapsed_seconds": round(res.elapsed_s, 1),
        **evaluate_model("transformer_simple", y_te, res.test_preds, tickers=tk_te,
                         bootstrap_n=args.bootstrap, rng_seed=seed or 0),
        "history": res.history,
    }
    with open(args.out / "transformer_simple_summary.json", "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o))

    pred_df = pd.DataFrame({
        "ticker":         tk_te,
        "date":           dt_te,
        "true_direction": y_te,
        "pred_transformer_simple": res.test_preds,
    })
    pred_df.to_csv(args.out / "transformer_simple_predictions.csv", index=False)

    pc = summary["per_class"]
    print("\n=== Transformer simple summary ===")
    print(f"  test_acc  = {summary['accuracy']*100:.2f}%   CI95 [{summary['ci95_lo']*100:.2f}, {summary['ci95_hi']*100:.2f}]")
    print(f"  F1_UP     = {pc['UP']['f1']:.4f}   F1_DOWN = {pc['DOWN']['f1']:.4f}   macro = {pc['macro_f1']:.4f}")
    print(f"  Wrote: {args.out / 'transformer_simple_summary.json'}")
    print(f"  Wrote: {args.out / 'transformer_simple_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
