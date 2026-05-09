"""training/baselines/_deep_common.py

Shared windowing / dataset / training-loop infrastructure for the
deep baselines of Action 1.3 (LSTM simple, Transformer standard).
The two model files import this module so the data path stays bit
identical and the LaTeX rows produced are directly comparable.

The module is deliberately self-contained and uses only PyTorch +
numpy + pandas. The metric computation (CIs, McNemar) is delegated
to training/baselines/metrics.py.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent

TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"
HORIZON = 7

FEATURES = [
    "close_price", "open_price", "high_price", "low_price",
    "volume", "daily_return", "ma_5", "ma_20", "ma_50",
    "rsi_14", "volatility_20", "volume_ma_20",
    "price_to_ma20", "rsi_momentum", "volume_spike",
]


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------

def set_seed_from_env(default: int | None = 0) -> int | None:
    seed_env = os.environ.get("BVMT_SEED")
    seed = int(seed_env) if seed_env is not None else default
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------

def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    if "price_to_ma20" not in df.columns:
        df["price_to_ma20"] = (df["close_price"] / df["ma_20"].replace(0, np.nan)).fillna(1.0).clip(0.5, 2.0).astype("float32")
    if "rsi_momentum" not in df.columns:
        df["rsi_momentum"] = (
            df.groupby("ticker")["rsi_14"]
              .transform(lambda x: x - x.shift(5))
              .fillna(0.0).astype("float32")
        )
    if "volume_spike" not in df.columns:
        df["volume_spike"] = (df["volume"] / df["volume_ma_20"].replace(0, np.nan)).fillna(1.0).clip(0.0, 10.0).astype("float32")
    return df


def fit_per_stock_stats(train_df: pd.DataFrame) -> dict:
    stats = {}
    for ticker, g in train_df.groupby("ticker"):
        s = {}
        for f in FEATURES:
            v = g[f].astype("float32")
            mu = float(v.mean())
            sd = float(v.std())
            if not np.isfinite(sd) or sd < 1e-8:
                sd = 1.0
            s[f] = (mu, sd)
        stats[str(ticker)] = s
    return stats


def apply_per_stock_norm(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    for f in FEATURES:
        mus = out["ticker"].map(lambda t: stats.get(str(t), {}).get(f, (0.0, 1.0))[0]).astype("float32")
        sds = out["ticker"].map(lambda t: stats.get(str(t), {}).get(f, (0.0, 1.0))[1]).astype("float32")
        out[f] = ((out[f].astype("float32") - mus) / sds).astype("float32")
    return out


def build_windows(
    df: pd.DataFrame,
    window: int,
    split_lo: pd.Timestamp,
    split_hi: pd.Timestamp,
    boundary_drop: bool = True,
):
    """3-D float32 array of shape [N, window, n_features], plus 1-D y, ticker, date arrays."""
    Xs, ys, tickers, dates = [], [], [], []
    n_feat = len(FEATURES)
    for ticker, g in df.groupby("ticker", sort=True):
        g = g.sort_values("time_idx").reset_index(drop=True)
        feats = g[FEATURES].astype("float32").to_numpy()
        labels = g["direction_7d"].astype(int).to_numpy()
        d = pd.to_datetime(g["date"]).to_numpy()
        n = len(g)
        if n < window + HORIZON:
            continue
        for t in range(window - 1, n):
            row_date = d[t]
            if not (split_lo <= row_date < split_hi):
                continue
            if boundary_drop:
                if t + HORIZON >= n or d[t + HORIZON] >= split_hi:
                    continue
            Xs.append(feats[t - window + 1: t + 1])
            ys.append(int(labels[t]))
            tickers.append(str(ticker))
            dates.append(np.datetime_as_string(row_date, unit="D"))
    if not Xs:
        return (
            np.empty((0, window, n_feat), dtype="float32"),
            np.empty(0, dtype=int),
            np.empty(0, dtype=object),
            np.empty(0, dtype=object),
        )
    return (
        np.stack(Xs).astype("float32"),
        np.asarray(ys, dtype=int),
        np.asarray(tickers, dtype=object),
        np.asarray(dates, dtype=object),
    )


# ----------------------------------------------------------------------
# Torch helpers
# ----------------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def class_weights(y: np.ndarray) -> torch.Tensor:
    n_total = len(y)
    n_up = max(int((y == 1).sum()), 1)
    n_dn = max(int((y == 0).sum()), 1)
    return torch.tensor([n_total / (2 * n_dn), n_total / (2 * n_up)], dtype=torch.float32)


@dataclass
class TrainResult:
    val_acc: float
    test_acc: float
    test_preds: np.ndarray
    history: list[dict]
    best_epoch: int
    elapsed_s: float


def train_classifier(
    model: torch.nn.Module,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    *,
    device: torch.device,
    max_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    log_prefix: str = "",
) -> TrainResult:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cw = class_weights(y_tr).to(device)
    crit = torch.nn.CrossEntropyLoss(weight=cw)

    train_loader = DataLoader(WindowDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(WindowDataset(X_va, y_va), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(WindowDataset(X_te, y_te), batch_size=batch_size, shuffle=False, num_workers=0)

    best_val = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    history = []
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        tr_loss_sum, tr_correct, tr_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss_sum += float(loss.item()) * yb.size(0)
            tr_correct += int((logits.argmax(dim=-1) == yb).sum().item())
            tr_n += yb.size(0)

        # Validation
        model.eval()
        va_correct, va_n = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                va_correct += int((logits.argmax(dim=-1) == yb).sum().item())
                va_n += yb.size(0)
        val_acc = va_correct / max(va_n, 1)
        train_acc = tr_correct / max(tr_n, 1)
        history.append({
            "epoch": ep, "train_loss": round(tr_loss_sum / max(tr_n, 1), 4),
            "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
        })
        elapsed = time.time() - t0
        improved = val_acc > best_val + 1e-6
        marker = " <-best" if improved else ""
        print(f"  {log_prefix}ep {ep:2d}/{max_epochs} train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  ({elapsed:.0f}s){marker}")
        if improved:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  {log_prefix}early stop @ epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    test_preds = []
    te_correct, te_n = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=-1).cpu().numpy()
            test_preds.extend(preds.tolist())
            te_correct += int((logits.argmax(dim=-1) == yb).sum().item())
            te_n += yb.size(0)
    test_acc = te_correct / max(te_n, 1)
    return TrainResult(
        val_acc=best_val, test_acc=test_acc,
        test_preds=np.asarray(test_preds, dtype=int),
        history=history, best_epoch=best_epoch,
        elapsed_s=time.time() - t0,
    )


def load_split(
    data_path: Path,
    window: int,
    smoke: int = 0,
):
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "time_idx"]).reset_index(drop=True)
    df = add_relative_features(df)
    if smoke > 0:
        keep = sorted(df["ticker"].unique())[:smoke]
        df = df[df["ticker"].isin(keep)].copy()
        print(f"  Smoke mode: {len(keep)} tickers ({keep})")

    train_hi = pd.Timestamp(TRAIN_END)
    val_hi = pd.Timestamp(VAL_END)
    test_hi = pd.Timestamp("2099-01-01")

    train_rows = df[df["date"] < train_hi].copy()
    print(f"  Fitting per-stock norm on {len(train_rows):,} train rows.")
    stats = fit_per_stock_stats(train_rows)
    df_norm = apply_per_stock_norm(df, stats)

    print(f"  Building windows (size={window})...")
    X_tr, y_tr, tk_tr, _ = build_windows(df_norm, window, pd.Timestamp("1900-01-01"), train_hi)
    X_va, y_va, tk_va, _ = build_windows(df_norm, window, train_hi, val_hi)
    X_te, y_te, tk_te, dt_te = build_windows(df_norm, window, val_hi, test_hi)
    print(f"  train: {X_tr.shape}  val: {X_va.shape}  test: {X_te.shape}")
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te, tk_te, dt_te)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
