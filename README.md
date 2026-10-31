# MGDSM

Manifold-Aware Graph-Diffused Semantic Model for Generalized Zero-Shot Fault Diagnosis.

## Code availability

**The complete source code will be released after the peer-review process is completed.**

This preliminary release contains only selected model components. It is not a
complete training or inference implementation and cannot reproduce the reported
experiments on its own.

## Included components

- `FeatureExtractor` implements the process-feature encoder.
- `GCNLayer` implements a graph convolution with a supplied normalized adjacency.

The full MGDSM model, domain-identification logic, score diffusion, losses,
training and evaluation pipelines, experiment configurations, and checkpoints
are not included in this preliminary release.

## Installation

Python 3.10 or later is required.

```bash
pip install -r requirements.txt
```

## Component example

```python
import torch
from mgdsm import FeatureExtractor, GCNLayer

encoder = FeatureExtractor()
features = encoder(torch.randn(4, 52))
assert features.shape == (4, 25)

layer = GCNLayer(in_dim=20, out_dim=25)
node_features = torch.randn(15, 20)
normalized_adjacency = torch.eye(15)
class_features = layer(node_features, normalized_adjacency)
assert class_features.shape == (15, 25)
```

The example checks component interfaces using random tensors. It is not an
experiment or a trained diagnostic model.
