"""Phase 1 Action 1.4 — TFT internal-ablation runner.

Three variants are exposed via the runner CLI:

  baseline    Reproduces train_tft_v3.py with controlled seed
              (the v3 reference re-trained from scratch).
  no_vsn      Variable Selection Network gating neutralized
              (uniform weights instead of softmax-gated weights).
  global_norm GroupNormalizer(groups=['ticker_id']) replaced by
              TorchNormalizer (single mu, sigma across the corpus).

A fourth ablation, --confidence-gating, is post-hoc and lives in
no_gating_eval.py because it operates on saved quantile predictions
rather than retraining.
"""
