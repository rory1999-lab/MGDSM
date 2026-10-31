"""Process and semantic encoders, Gaussian class weights, and auxiliary heads.

This module exposes the neural forward computation, not the complete MGDSM
domain-identification or fault-diagnosis pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .distributions import sample_class_weights
from .graph_utils import cosine_adjacency, normalize_adjacency


class FeatureExtractor(nn.Sequential):
    """Process-feature encoder extracted from the MGDSM architecture."""

    def __init__(self, in_dim: int = 52, embed_dim: int = 25, hidden_dim: int = 32):
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )


class GCNLayer(nn.Module):
    """Graph convolution with a caller-supplied normalized adjacency."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm @ x @ self.weight


@dataclass
class GaussianEncoderOutput:
    features: torch.Tensor
    mean: torch.Tensor
    log_variance: torch.Tensor


class ProcessEncoder(nn.Module):
    """Encode process samples and parameterize their diagonal Gaussians."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(
            in_dim=config.input_dim,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
        )
        self.mean_head = nn.Linear(config.embed_dim, config.embed_dim)
        self.logvar_head = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x: torch.Tensor) -> GaussianEncoderOutput:
        features = self.feature_extractor(x)
        return GaussianEncoderOutput(
            features=features,
            mean=self.mean_head(features),
            log_variance=self.logvar_head(features).clamp(
                self.config.logvar_min, self.config.logvar_max
            ),
        )


class SemanticEncoder(nn.Module):
    """Apply two graph convolutions and parameterize class Gaussians."""

    def __init__(self, attr_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gcn1 = GCNLayer(attr_dim, config.embed_dim)
        self.gcn2 = GCNLayer(config.embed_dim, config.embed_dim)
        self.mean_head = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.logvar_head = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

    def forward(
        self, attributes: torch.Tensor, normalized_adjacency: torch.Tensor
    ) -> GaussianEncoderOutput:
        hidden = F.leaky_relu(self.gcn1(attributes, normalized_adjacency), 0.2)
        features = F.leaky_relu(self.gcn2(hidden, normalized_adjacency), 0.2)
        return GaussianEncoderOutput(
            features=features,
            mean=self.mean_head(features),
            log_variance=self.logvar_head(features).clamp(
                self.config.logvar_min, self.config.logvar_max
            ),
        )


@dataclass
class BackboneOutput:
    """Raw tensors from the neural modules, without a domain decision."""

    features: torch.Tensor
    sample_mean: torch.Tensor
    sample_log_variance: torch.Tensor
    semantic_features: torch.Tensor
    class_mean: torch.Tensor
    class_log_variance: torch.Tensor
    classifier_samples: torch.Tensor
    logits_samples: torch.Tensor
    attribute_logits: torch.Tensor
    reconstruction: torch.Tensor


class MGDSMBackbone(nn.Module):
    """Compose the selected neural components into a forward model.

    Attributes are a [C, A] NumPy matrix. Inputs are standardized [B, D_x]
    tensors. Preprocessing statistics must be supplied by the caller.
    Gaussian class weights are sampled in both train and evaluation modes.
    No domain threshold, graph-score diffusion, or final fault label is computed.
    """

    def __init__(self, attributes: np.ndarray, config: ModelConfig | None = None):
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        attributes = np.asarray(attributes, dtype=np.float32)
        adjacency = cosine_adjacency(
            attributes, add_self_loops=True, top_k=self.config.graph_top_k
        )
        self.register_buffer("attributes", torch.tensor(attributes))
        self.register_buffer(
            "normalized_adjacency", torch.tensor(normalize_adjacency(adjacency))
        )
        self.process_encoder = ProcessEncoder(self.config)
        self.semantic_encoder = SemanticEncoder(attributes.shape[1], self.config)
        self.attribute_head = nn.Linear(
            self.config.input_dim + self.config.embed_dim, attributes.shape[1]
        )
        self.reconstruction_head = nn.Linear(self.config.embed_dim, self.config.input_dim)

    def forward(
        self,
        x: torch.Tensor,
        num_draws: int = 1,
        generator: torch.Generator | None = None,
    ) -> BackboneOutput:
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.config.input_dim:
            raise ValueError("x must have nonempty [batch, input_dim] shape")
        process = self.process_encoder(x)
        semantic = self.semantic_encoder(self.attributes, self.normalized_adjacency)
        classifiers = sample_class_weights(
            semantic.mean, semantic.log_variance, num_draws, generator=generator
        )
        # Classification and auxiliary heads use deterministic process features.
        logits = torch.einsum("bd,rcd->rbc", process.features, classifiers)
        attribute_input = torch.cat((x, process.features), dim=1)
        return BackboneOutput(
            features=process.features,
            sample_mean=process.mean,
            sample_log_variance=process.log_variance,
            semantic_features=semantic.features,
            class_mean=semantic.mean,
            class_log_variance=semantic.log_variance,
            classifier_samples=classifiers,
            logits_samples=logits,
            attribute_logits=self.attribute_head(attribute_input),
            reconstruction=self.reconstruction_head(process.features),
        )
