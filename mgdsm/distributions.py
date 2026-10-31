"""Sampling and probability aggregation for Gaussian class weights."""
from __future__ import annotations

import torch


def sample_class_weights(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    num_draws: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample weights with shape [draws, classes, embedding dimension]."""
    if mean.ndim != 2 or mean.shape != log_variance.shape or min(mean.shape) == 0:
        raise ValueError("mean and log_variance must have matching nonempty [C, D] shapes")
    if not isinstance(num_draws, int) or isinstance(num_draws, bool) or num_draws < 1:
        raise ValueError("num_draws must be a positive integer")
    noise = torch.randn(
        num_draws, *mean.shape, device=mean.device, dtype=mean.dtype, generator=generator
    )
    return mean.unsqueeze(0) + torch.exp(0.5 * log_variance).unsqueeze(0) * noise


def mean_class_probabilities(logits_samples: torch.Tensor) -> torch.Tensor:
    """Average softmax probabilities over draws, returning shape [B, C].

    The caller defines the candidate classes before calling this function.
    This function does not select the seen or unseen domain.
    """
    if logits_samples.ndim != 3 or min(logits_samples.shape) == 0:
        raise ValueError("logits_samples must have nonempty [draws, batch, classes] shape")
    return logits_samples.softmax(dim=-1).mean(dim=0)
