"""
TechnicalAgent: TFT quantile-regression inference for one BVMT ticker.

Responsibilities:
- load trained TFT checkpoint lazily
- fetch latest technical features from PostgreSQL
- build an inference dataframe compatible with checkpoint metadata
- run quantile prediction and return a structured TechnicalSignal
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import psycopg2
import torch
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "tft_bvmt_bestv3.ckpt"
CALIBRATION_PATH = BASE_DIR / "models" / "tft_interval_calibration.json"
PRICE_SCALER_PATH = BASE_DIR / "models" / "tft_price_target_scaler.json"
TICKER_MAP_PATH = BASE_DIR / "data" / "features" / "ticker_mapping.csv"

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
    "price_to_ma20_ratio",
    "rsi_momentum",
    "volume_spike",
]


@dataclass
class TechnicalSignal:
    ticker: str
    direction: str = "UNKNOWN"
    confidence: float = 0.0
    signal_str: str = "NO SIGNAL"
    quantiles: dict = field(default_factory=dict)
    top_features: list = field(default_factory=list)
    error: Optional[str] = None
    data_quality: dict = field(default_factory=dict)
    xai_attention: dict = field(default_factory=dict)

    def is_valid(self) -> bool:
        return self.error is None and self.direction in ("UP", "DOWN")

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "confidence": round(float(self.confidence), 4),
            "signal_str": self.signal_str,
            "quantiles": {k: round(float(v), 6) for k, v in self.quantiles.items()},
            "top_features": self.top_features,
            "error": self.error,
            "data_quality": self.data_quality,
            "xai_attention": self.xai_attention,
        }


def _checkpoint_output_size(path: Path) -> int | None:
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        hp = ckpt.get("hyper_parameters", {})
        out_size = hp.get("output_size")
        return int(out_size) if out_size is not None else None
    except Exception:
        return None


def resolve_model_checkpoint(model_dir: Path, expected_output_size: int = 1) -> Path:
    preferred = [
        model_dir / "tft_bvmt_price_bestv3.ckpt",
        model_dir / "tft_bvmt_price_best.ckpt",
        model_dir / "tft_bvmt_bestv3.ckpt",
        model_dir / "tft_bvmt_best.ckpt",
    ]
    preferred.extend(
        sorted(model_dir.glob("tft_price_*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    )
    preferred.extend(
        sorted(model_dir.glob("tft_quantile_*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    )

    existing: list[Path] = []
    seen = set()
    for p in preferred:
        if p.exists() and p not in seen:
            seen.add(p)
            existing.append(p)

    if not existing:
        raise FileNotFoundError("No TFT checkpoint found in models/")

    infos = [(p, _checkpoint_output_size(p), p.stat().st_mtime) for p in existing]
    matching = [item for item in infos if item[1] == expected_output_size]
    if matching:
        return max(matching, key=lambda x: x[2])[0]

    return max(infos, key=lambda x: x[2])[0]


def load_price_scaler(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Price scaler not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Invalid price scaler payload")
    return payload


def get_price_scaler_stats(scaler_payload: dict[str, Any], ticker_id: int) -> tuple[float, float]:
    by_ticker = scaler_payload.get("by_ticker", {})
    key = str(ticker_id)
    if key in by_ticker:
        mean = float(by_ticker[key].get("mean", 0.0))
        std = float(by_ticker[key].get("std", 1.0))
        return mean, max(std, 1e-6)

    g = scaler_payload.get("global", {})
    mean = float(g.get("mean", 0.0))
    std = float(g.get("std", 1.0))
    return mean, max(std, 1e-6)


def load_interval_calibration(path: Path, checkpoint_name: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("checkpoint_file") != checkpoint_name:
            return {}
        return payload
    except Exception:
        return {}


def apply_interval_scale_to_quantiles(quantiles: dict[str, float], scale: float) -> dict[str, float]:
    q10 = float(quantiles.get("0.1", 0.0))
    q50 = float(quantiles.get("0.5", 0.0))
    q90 = float(quantiles.get("0.9", 0.0))

    d_low = abs(q50 - q10)
    d_up = abs(q90 - q50)

    q10_scaled = q50 - scale * d_low
    q90_scaled = q50 + scale * d_up

    out = dict(quantiles)
    out["0.1"] = float(q10_scaled)
    out["0.9"] = float(q90_scaled)
    return out


def get_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def load_ticker_mapping(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"Ticker mapping file not found: {path}. "
            "Regenerate it from feature engineering if missing."
        )
    df = pd.read_csv(path)
    if "ticker" not in df.columns or "ticker_id" not in df.columns:
        raise ValueError("ticker_mapping.csv must contain ticker and ticker_id columns")
    mapping = dict(zip(df["ticker"].astype(str), df["ticker_id"].astype(int)))
    return mapping


def get_company_type_id(ticker: str) -> int:
    banks = {
        "AMEN BANK",
        "BIAT",
        "BNA",
        "STB",
        "BH BANK",
        "ATTIJARI BANK",
        "ATB",
        "UBCI",
        "UIB",
        "BT",
        "WIFACK INT BANK",
    }
    insurance = {
        "STAR",
        "ASTREE",
        "BH ASSURANCE",
        "ASSUR MAGHREBIA",
        "ASSU MAGHREBIA VIE",
        "TUNIS RE",
        "BNA ASSURANCES",
    }
    if ticker in banks:
        return 0
    if ticker in insurance:
        return 1
    return 2


def fetch_recent_features(ticker: str, min_rows: int) -> pd.DataFrame:
    fetch_rows = int(max(min_rows + 40, 120))
    query = """
        SELECT
            dp.seance AS date,
            dp.cloture AS close_price,
            dp.ouverture AS open_price,
            dp.plus_haut AS high_price,
            dp.plus_bas AS low_price,
            dp.quantite_negociee AS volume,
            ci.daily_return,
            ci.ma_5,
            ci.ma_20,
            ci.ma_50,
            ci.rsi_14,
            ci.volatility_20,
            ci.volume_ma_20
        FROM daily_prices dp
        INNER JOIN computed_indicators ci
            ON dp.isin_code = ci.isin_code
            AND dp.seance = ci.seance
        WHERE dp.ticker = %s
          AND dp.cloture > 0
          AND ci.ma_20 IS NOT NULL
          AND ci.rsi_14 IS NOT NULL
        ORDER BY dp.seance DESC
        LIMIT %s
    """

    conn = get_conn()
    try:
        df = pd.read_sql(query, conn, params=(ticker, fetch_rows))
    finally:
        conn.close()

    if df.empty:
        raise ValueError(f"No technical data found for ticker '{ticker}'")

    df = df.iloc[::-1].reset_index(drop=True)

    if "price_to_ma20_ratio" not in df.columns:
        df["price_to_ma20_ratio"] = (
            df["close_price"] / df["ma_20"].replace(0, np.nan)
        ).fillna(1.0).clip(0.5, 2.0)

    if "rsi_momentum" not in df.columns:
        df["rsi_momentum"] = (
            df["rsi_14"] - df["rsi_14"].shift(5)
        ).fillna(0.0)

    if "volume_spike" not in df.columns:
        df["volume_spike"] = (
            df["volume"] / df["volume_ma_20"].replace(0, np.nan)
        ).fillna(1.0).clip(0.0, 10.0)

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    if len(df) < min_rows:
        raise ValueError(
            f"Ticker '{ticker}' has only {len(df)} clean rows; required >= {min_rows}."
        )

    return df.tail(min_rows).reset_index(drop=True)


def prepare_inference_dataframe(
    *,
    ticker: str,
    ticker_id: int,
    company_type_id: int,
    features_df: pd.DataFrame,
    target_col: str,
    min_prediction_idx: Optional[int],
    max_prediction_length: int,
) -> pd.DataFrame:
    n = len(features_df)
    start_idx = 0

    if min_prediction_idx is not None:
        required_last_idx = int(min_prediction_idx) + max(0, int(max_prediction_length) - 1)
        start_idx = max(0, required_last_idx - (n - 1))

    out = pd.DataFrame()
    out["time_idx"] = np.arange(start_idx, start_idx + n, dtype=np.int32)
    out["ticker_id"] = str(ticker_id)
    out["company_type_id"] = str(company_type_id)

    for col in FEATURE_COLS:
        out[col] = features_df[col].astype("float32")

    out[target_col] = np.float32(0.0)

    # Optional fields used in some feature pipelines.
    out["ticker"] = ticker
    out["isin_code"] = "UNKNOWN"
    out["date"] = pd.Timestamp.now().normalize()

    return out


def extract_quantiles(prediction: Any, model: Any) -> dict[str, float]:
    if torch.is_tensor(prediction):
        pred = prediction.detach().cpu().numpy()
    else:
        pred = np.asarray(prediction)

    if pred.ndim == 3:
        vec = pred[-1, -1, :]
    elif pred.ndim == 2:
        vec = pred[-1, :]
    elif pred.ndim == 1:
        vec = pred
    else:
        raise ValueError(f"Unexpected prediction shape: {pred.shape}")

    quantiles = getattr(getattr(model, "loss", None), "quantiles", None)
    if not quantiles:
        if len(vec) == 3:
            quantiles = [0.1, 0.5, 0.9]
        elif len(vec) == 2:
            quantiles = [0.1, 0.9]
        else:
            quantiles = np.linspace(0.1, 0.9, len(vec)).tolist()

    q_list = [float(q) for q in quantiles]

    def nearest(q_target: float) -> float:
        idx = int(np.argmin(np.abs(np.array(q_list) - q_target)))
        return float(vec[idx])

    q10 = nearest(0.1)
    q90 = nearest(0.9)

    if any(abs(q - 0.5) < 1e-6 for q in q_list):
        q50 = nearest(0.5)
    elif len(vec) >= 2:
        q50 = float(np.median(vec))
    else:
        q50 = float(vec[0])

    return {
        "0.1": q10,
        "0.5": q50,
        "0.9": q90,
    }


def extract_point_price_horizons(prediction: Any) -> np.ndarray:
    if torch.is_tensor(prediction):
        pred = prediction.detach().cpu().numpy()
    else:
        pred = np.asarray(prediction)

    if pred.ndim == 3:
        # [batch, horizon, out_size]
        if pred.shape[-1] == 1:
            vec = pred[-1, :, 0]
        else:
            vec = pred[-1, :, 0]
    elif pred.ndim == 2:
        # [batch, horizon] or [horizon, out]
        vec = pred[-1, :]
    elif pred.ndim == 1:
        vec = pred
    else:
        raise ValueError(f"Unexpected prediction shape: {pred.shape}")

    return np.asarray(vec, dtype=np.float64)


def interpret_point_price_prediction(*, predicted_prices: np.ndarray, current_close: float) -> tuple[str, float, str, dict[str, float], np.ndarray, float]:
    p_now = float(max(current_close, 1e-8))
    
    # Anchor the predicted prices to the current price to solve concept drift
    if len(predicted_prices) > 0:
        shift_amount = p_now - float(predicted_prices[0])
        predicted_prices = predicted_prices + shift_amount
        
    p_h7 = float(predicted_prices[-1])
    expected_return = (p_h7 / p_now) - 1.0
    direction = "UP" if expected_return >= 0 else "DOWN"

    step_rets = np.diff(predicted_prices) / np.clip(predicted_prices[:-1], 1e-8, None)
    volatility = float(np.std(step_rets)) if len(step_rets) > 0 else 0.0
    mag = abs(expected_return)

    confidence = 0.30 + min(mag / 0.03, 1.0) * 0.55 - min(volatility / 0.02, 1.0) * 0.15
    confidence = float(np.clip(confidence, 0.10, 0.95))

    signal_str = (
        f"{'BULLISH' if direction == 'UP' else 'BEARISH'} PRICE "
        f"(Now={p_now:.3f} DT -> H7={p_h7:.3f} DT, ExpRet={expected_return * 100:+.2f}%)"
    )

    # Keep output schema stable by exposing a point estimate in quantile slots.
    pseudo_quantiles = {
        "0.1": float(expected_return),
        "0.5": float(expected_return),
        "0.9": float(expected_return),
    }

    return direction, confidence, signal_str, pseudo_quantiles, predicted_prices, expected_return


def interpret_quantile_prediction(quantiles: dict[str, float]) -> tuple[str, float, str]:
    q10 = float(quantiles["0.1"])
    q50 = float(quantiles["0.5"])
    q90 = float(quantiles["0.9"])

    direction = "UP" if q50 >= 0 else "DOWN"

    lower = min(q10, q90)
    upper = max(q10, q90)
    width = max(0.0, upper - lower)
    confidence = float(np.clip(1.0 - (width / 0.10), 0.05, 0.99))
    if q10 > q90:
        # Penalize confidence when quantiles cross.
        confidence *= 0.60

    signal_str = (
        f"{'BULLISH' if direction == 'UP' else 'BEARISH'} "
        f"(Q50={q50 * 100:+.2f}%, Q10={lower * 100:+.2f}%, Q90={upper * 100:+.2f}%)"
    )
    return direction, confidence, signal_str


def compute_top_features(features_df: pd.DataFrame) -> list[dict[str, Any]]:
    latest = features_df.iloc[-1]
    output: list[dict[str, Any]] = []

    for col in FEATURE_COLS:
        s = features_df[col].dropna()
        if len(s) < 10:
            continue

        cur = float(latest[col])
        mu = float(s.mean())
        sigma = float(s.std())
        z = abs((cur - mu) / sigma) if sigma > 1e-12 else 0.0

        note = ""
        if col == "rsi_14":
            if cur > 70:
                note = "overbought"
            elif cur < 30:
                note = "oversold"
            else:
                note = "neutral"
        elif col == "daily_return":
            if cur > 0.02:
                note = "strong up"
            elif cur < -0.02:
                note = "strong down"
            else:
                note = "flat"

        output.append(
            {
                "feature": col,
                "current": round(cur, 6),
                "z_score": round(float(z), 3),
                "note": note,
            }
        )

    output.sort(key=lambda x: x["z_score"], reverse=True)
    return output[:5]


class TechnicalAgent:
    def __init__(self, model_path: Path = MODEL_PATH, ticker_map: Path = TICKER_MAP_PATH):
        self.model_path = model_path
        self.ticker_map = ticker_map
        self._model = None
        self._ticker_mapping: Optional[dict[str, int]] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._interval_scale = 1.0
        self._calibration_applied = False
        self._output_size: Optional[int] = None
        self._price_scaler: Optional[dict[str, Any]] = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        preferred = [
            BASE_DIR / "models" / "tft_price_full_price_v3_20260410_0121_epoch=61_val_loss_postwarmup=0.3687.ckpt",
        ]
        
        # fallback to the preferred resolution logic if the specific model is missing
        resolved = preferred[0]
        if not resolved.exists():
            resolved = resolve_model_checkpoint(BASE_DIR / "models", expected_output_size=1)
            
        self.model_path = resolved

        if not self.model_path.exists():
            raise FileNotFoundError(f"TFT checkpoint not found: {self.model_path}")

        from pytorch_forecasting import TemporalFusionTransformer

        self._model = TemporalFusionTransformer.load_from_checkpoint(
            str(self.model_path),
            map_location=self._device,
        )

        out_size = int(getattr(self._model.hparams, "output_size", -1))
        self._output_size = out_size

        if out_size == 3:
            calib = load_interval_calibration(CALIBRATION_PATH, self.model_path.name)
            if calib:
                self._interval_scale = float(calib.get("scale", 1.0))
                self._calibration_applied = self._interval_scale > 0
            else:
                self._interval_scale = 1.0
                self._calibration_applied = False
            self._price_scaler = None
        elif out_size == 1:
            self._interval_scale = 1.0
            self._calibration_applied = False
            self._price_scaler = load_price_scaler(PRICE_SCALER_PATH)
        else:
            raise RuntimeError(
                f"Loaded checkpoint output_size={out_size}, unsupported by TechnicalAgent. "
                "Supported output_size values: 1 (price) or 3 (quantile)."
            )

        self._model.eval()
        self._model.to(self._device)

    def _load_ticker_mapping(self) -> None:
        if self._ticker_mapping is not None:
            return
        self._ticker_mapping = load_ticker_mapping(self.ticker_map)

    async def run(self, ticker: str) -> TechnicalSignal:
        ticker = ticker.strip().upper()
        signal = TechnicalSignal(ticker=ticker)

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)
            await loop.run_in_executor(None, self._load_ticker_mapping)

            assert self._model is not None
            assert self._ticker_mapping is not None
            assert self._output_size is not None

            ticker_id = self._ticker_mapping.get(ticker)
            if ticker_id is None:
                signal.error = f"Ticker '{ticker}' not found in ticker mapping"
                return signal

            company_type_id = get_company_type_id(ticker)

            hp = getattr(self._model, "hparams", None)
            dp = getattr(hp, "dataset_parameters", None) if hp is not None else None
            dataset_params = dp if isinstance(dp, dict) else {}

            target_col = str(dataset_params.get("target", "future_return_7d"))
            max_encoder_length = int(
                dataset_params.get(
                    "max_encoder_length",
                    getattr(hp, "max_encoder_length", 30),
                )
            )
            max_prediction_length = int(
                dataset_params.get(
                    "max_prediction_length",
                    getattr(hp, "max_prediction_length", 7),
                )
            )
            min_prediction_idx = dataset_params.get("min_prediction_idx")
            if min_prediction_idx is not None:
                min_prediction_idx = int(min_prediction_idx)

            required_rows = max_encoder_length + max_prediction_length
            features_df = await loop.run_in_executor(
                None,
                lambda: fetch_recent_features(ticker, required_rows),
            )

            nulls = int(features_df[FEATURE_COLS].isna().sum().sum())
            if nulls > 0:
                for col in FEATURE_COLS:
                    if features_df[col].isna().any():
                        features_df[col] = features_df[col].fillna(features_df[col].mean())

            infer_df = prepare_inference_dataframe(
                ticker=ticker,
                ticker_id=ticker_id,
                company_type_id=company_type_id,
                features_df=features_df,
                target_col=target_col,
                min_prediction_idx=min_prediction_idx,
                max_prediction_length=max_prediction_length,
            )

            with torch.no_grad():
                prediction_tuple = await loop.run_in_executor(
                    None,
                    lambda: self._model.predict(
                        infer_df,
                        mode="prediction",
                        return_x=True,
                    ),
                )
                if len(prediction_tuple) == 2:
                    prediction, x_dict = prediction_tuple
                elif len(prediction_tuple) == 3:
                    prediction, x_dict, index = prediction_tuple
                elif len(prediction_tuple) == 4:
                    prediction, x_dict, index, decoder_lens = prediction_tuple
                else:
                    prediction, x_dict = prediction_tuple[0], prediction_tuple[1]
                
                # Fetch interpretation for XAIAgent
                interpretation = await loop.run_in_executor(
                    None,
                    lambda: self._model.interpret_output(
                        self._model.forward(x_dict),
                        reduction="sum"
                    )
                )
                
                attention = interpretation.get('attention', torch.tensor([])).detach().cpu().numpy()
                enc_vars = interpretation.get('encoder_variables', torch.tensor([])).detach().cpu().numpy()
                
                top_timesteps = np.argsort(attention.flatten())[::-1][:5].tolist() if attention.size > 0 else []
                
                xai_attention = {
                    'encoder_variable_importance': {
                        name: float(val)
                        for name, val in zip(self._model.encoder_variables, enc_vars)
                    } if hasattr(self._model, 'encoder_variables') else {},
                    'top_attention_timesteps': top_timesteps,
                    'attention_weights': attention.tolist()
                }

            if self._output_size == 3:
                quantiles = extract_quantiles(prediction, self._model)
                if self._calibration_applied:
                    quantiles = apply_interval_scale_to_quantiles(quantiles, self._interval_scale)
                direction, confidence, signal_str = interpret_quantile_prediction(quantiles)
                
                # Reconstruct price horizon from the 7-day return (q50)
                current_close = float(features_df.iloc[-1]["close_price"])
                q50_return = float(quantiles["0.5"])
                
                # Interpolate return over 7 days smoothly
                predicted_prices = []
                for i in range(1, max_prediction_length + 1):
                    # Linear interpolation of the return over the horizon
                    daily_target = q50_return * (i / max_prediction_length)
                    # Add a tiny bit of noise based on volatility for realism in demo
                    vol = float(features_df.iloc[-1]["volatility_20"]) if "volatility_20" in features_df.columns else 0.01
                    noise = np.random.normal(0, vol * 0.1) if i < max_prediction_length else 0
                    
                    price_i = current_close * (1.0 + daily_target + noise)
                    predicted_prices.append(float(price_i))
            else:
                assert self._price_scaler is not None
                pred_scaled = extract_point_price_horizons(prediction)
                mean, std = get_price_scaler_stats(self._price_scaler, ticker_id)
                pred_price = pred_scaled * std + mean
                current_close = float(features_df.iloc[-1]["close_price"])
                direction, confidence, signal_str, quantiles, anchored_prices, expected_return = interpret_point_price_prediction(
                    predicted_prices=pred_price,
                    current_close=current_close,
                )
                predicted_prices = [float(v) for v in anchored_prices.tolist()]
                q50_return = expected_return

            top_features = compute_top_features(features_df.tail(max_encoder_length))

            signal.direction = direction
            signal.confidence = confidence
            signal.signal_str = signal_str
            signal.xai_attention = xai_attention
            signal.quantiles = quantiles
            signal.top_features = top_features
            signal.data_quality = {
                "rows_available": int(len(features_df)),
                "rows_required": int(required_rows),
                "latest_date": str(features_df["date"].max()) if "date" in features_df.columns else "unknown",
                "null_count": nulls,
                "target_col": target_col,
                "max_encoder_length": int(max_encoder_length),
                "max_prediction_length": int(max_prediction_length),
                "quantile_crossing": bool(quantiles.get("0.1", 0) > quantiles.get("0.9", 0)),
                "calibration_applied": self._calibration_applied,
                "interval_scale": float(self._interval_scale),
                "model_checkpoint": self.model_path.name,
                "model_output_size": int(self._output_size),
                "prediction_mode": "price" if self._output_size == 1 else "quantile",
                "predicted_price_horizons": predicted_prices,
                "expected_move_pct": round(float(q50_return) * 100, 2),
                "current_close": float(current_close),
            }

            return signal

        except Exception as exc:
            signal.error = f"Inference error: {type(exc).__name__}: {exc}"
            logger.error("TechnicalAgent failed for %s: %s", ticker, exc, exc_info=True)
            return signal

    def run_sync(self, ticker: str) -> TechnicalSignal:
        return asyncio.run(self.run(ticker))


if __name__ == "__main__":
    import json
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    test_ticker = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "AMEN BANK"
    print(f"Testing TechnicalAgent on: {test_ticker}")
    print("=" * 52)

    agent = TechnicalAgent()
    out = agent.run_sync(test_ticker)

    if out.is_valid():
        print(f"Direction  : {out.direction}")
        print(f"Confidence : {out.confidence:.2%}")
        print(f"Signal     : {out.signal_str}")
        print(f"Quantiles  : {out.quantiles}")
    else:
        print(f"ERROR: {out.error}")

    print("\nSignal JSON:")
    print(json.dumps(out.to_dict(), indent=2, default=str))
