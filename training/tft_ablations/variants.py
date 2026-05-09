"""training/tft_ablations/variants.py

Implements the two TRAINING-time TFT ablations of Action 1.4:

  - no_vsn:      neutralize the Variable Selection Network gating by
                 forcing uniform per-variable weights. The single-variable
                 GRNs still run; only the softmax-gated weighting that
                 multiplies them is replaced by 1/N.
  - global_norm: replace the per-stock GroupNormalizer with a global
                 TorchNormalizer fitted across all 68 tickers. This
                 collapses the per-stock target standardisation that the
                 v3 article identifies as one of the keys to lifting
                 directional accuracy on heterogeneous BVMT scales.

Why these specific changes? The improvement plan (paper/improvement_plan.md
Action 1.4) names exactly these three components as candidates for the +24-25
pp gap between TFT v3 and the 50-53% baselines: VSN gating, per-stock
normalisation, and confidence gating (the latter is post-hoc, in
no_gating_eval.py). If both no_vsn and global_norm degrade strongly while
the baseline holds at 77.4%, the gap is mechanism-attributable, not
data-luck.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from pytorch_forecasting.data import GroupNormalizer, TorchNormalizer


def make_target_normalizer(variant: str):
    """Return the TimeSeriesDataSet target normalizer for the requested variant.

    baseline / no_vsn -> GroupNormalizer(groups=['ticker_id'])
        Per-stock z-score on the training fraction. This is what
        train_tft_v3.py uses (post-F1 fix). Symmetric, no transformation.

    global_norm -> TorchNormalizer()
        Single (mu, sigma) computed over the entire training corpus.
        AMEN BANK and a thinly traded mid-cap share the same scale; this
        is the *naive* baseline against which the per-stock approach is
        meant to demonstrate gain.
    """
    if variant in ("baseline", "no_vsn"):
        return GroupNormalizer(groups=["ticker_id"], transformation=None)
    if variant == "global_norm":
        return TorchNormalizer(method="standard", center=True)
    raise ValueError(f"unknown variant: {variant}")


# -----------------------------------------------------------------------------
# no_vsn: monkey-patch each VariableSelectionNetwork inside an instantiated TFT
# so that the per-variable softmax gating is replaced by uniform 1/N weights.
#
# The GRNs themselves are kept: the ablation isolates the *gating*
# contribution, not the GRN contribution. A more aggressive "no GRN" ablation
# would essentially be the standard Transformer baseline already trained in
# Action 1.3 (52.26%), so it would not add information.
# -----------------------------------------------------------------------------

def _uniform_vsn_forward(self, x, context=None):
    """Replacement forward for VariableSelectionNetwork.

    Mirrors the upstream forward in pytorch_forecasting (v1.7.0,
    sub_modules.VariableSelectionNetwork.forward) except that
    ``sparse_weights`` are forced uniform instead of being produced by
    ``softmax(flattened_grn(concat(x)))``.
    """
    if self.num_inputs > 1:
        var_outputs = []
        for name in self.input_sizes.keys():
            variable_embedding = x[name]
            if name in self.prescalers:
                variable_embedding = self.prescalers[name](variable_embedding)
            var_outputs.append(self.single_variable_grns[name](variable_embedding))
        var_outputs = torch.stack(var_outputs, dim=-1)

        n_var = var_outputs.size(-1)
        # Uniform gating: 1/N along the variable axis. Matches the
        # original sparse_weights shape: [..., 1, n_var] (the unsqueeze(-2)
        # adds the broadcast axis over hidden_size).
        uniform = torch.full_like(
            var_outputs[..., :1, :], 1.0 / n_var
        )
        outputs = var_outputs * uniform
        outputs = outputs.sum(dim=-1)
        return outputs, uniform.detach()

    if self.num_inputs == 1:
        # Single-input path — original VSN already does no gating here, so
        # the no_vsn variant behaves identically to the baseline. Replicate
        # the upstream code so callers receive a tuple of the expected shape.
        name = next(iter(self.single_variable_grns.keys()))
        variable_embedding = x[name]
        if name in self.prescalers:
            variable_embedding = self.prescalers[name](variable_embedding)
        outputs = self.single_variable_grns[name](variable_embedding)
        if outputs.ndim == 3:
            sparse_weights = torch.ones(
                outputs.size(0), outputs.size(1), 1, 1, device=outputs.device
            )
        else:
            sparse_weights = torch.ones(
                outputs.size(0), 1, 1, device=outputs.device
            )
        return outputs, sparse_weights

    # zero-input path — also unchanged
    outputs = torch.zeros(context.size(), device=context.device)
    if outputs.ndim == 3:
        sparse_weights = torch.zeros(
            outputs.size(0), outputs.size(1), 1, 0, device=outputs.device
        )
    else:
        sparse_weights = torch.zeros(outputs.size(0), 1, 0, device=outputs.device)
    return outputs, sparse_weights


def apply_no_vsn_patch(tft: nn.Module) -> Tuple[int, list[str]]:
    """Patch every VariableSelectionNetwork inside ``tft`` to use uniform gating.

    Walks the module tree and replaces ``forward`` on each VSN instance.
    Returns ``(n_patched, names)`` so callers can log a sanity line and
    abort if the count looks wrong. v3 has three VSNs (static, encoder,
    decoder).
    """
    from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
        VariableSelectionNetwork,
    )

    patched: list[str] = []
    for name, module in tft.named_modules():
        if isinstance(module, VariableSelectionNetwork):
            module.forward = _uniform_vsn_forward.__get__(module, type(module))
            patched.append(name or "<root>")
    return len(patched), patched
