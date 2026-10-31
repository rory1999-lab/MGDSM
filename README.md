# MGDSM

Manifold-Aware Graph-Diffused Semantic Model for Generalized Zero-Shot Fault Diagnosis.

## Code availability

**The complete source code will be released after the peer-review process is completed.**

This preliminary release provides the neural modeling components and a runnable
forward demonstration. The full training and fault-diagnosis pipelines will be
part of the complete release.

## Included model components

| Component | Function |
| --- | --- |
| `FeatureExtractor` | Maps process observations to deterministic latent features |
| `ProcessEncoder` | Produces process features and sample Gaussian parameters |
| `SemanticEncoder` | Uses a two-layer GCN to produce class Gaussian parameters |
| `sample_class_weights` | Samples classifier weights from class Gaussians |
| `MGDSMBackbone` | Combines the encoders, sampled classifier, and auxiliary heads |
| `cosine_adjacency` | Constructs the symmetric attribute graph |
| `normalize_adjacency` | Applies symmetric degree normalization |
| `mean_class_probabilities` | Averages conditional class probabilities over draws |

The attribute head uses the concatenation of the input and deterministic process
features. The reconstruction head also uses the deterministic features. The
sample Gaussian heads parameterize a distribution without replacing these
features in the classifier or auxiliary heads.

The release includes graph construction and GCN aggregation. GDSS score
diffusion, domain routing, threshold calibration, losses, training, dataset
preparation, experimental predictions, and trained checkpoints are reserved for
the complete release. Raw logits from the backbone are not final GZFD predictions.

## Install and run

Python 3.10 or later is required.

```bash
pip install -r requirements.txt
python demo_forward.py
python -m unittest discover -s tests -v
```

The demonstration uses random process inputs and a synthetic attribute matrix.
It prints tensor shapes without producing experimental accuracy results.

## Use the forward model

```python
import numpy as np
import torch
from mgdsm import MGDSMBackbone, ModelConfig, mean_class_probabilities

attributes = np.random.default_rng(0).integers(0, 2, size=(15, 20))
model = MGDSMBackbone(attributes, ModelConfig()).eval()
x = torch.randn(4, 52)

with torch.no_grad():
    output = model(x, num_draws=3)

# Select candidate columns before softmax when evaluating a conditional branch.
candidate_columns = torch.tensor([0, 1, 2])
probabilities = mean_class_probabilities(output.logits_samples[:, :, candidate_columns])
assert probabilities.shape == (4, 3)
```

Inputs to the backbone must already be standardized. The synthetic attributes
above are for interface demonstration and are not the fault attribute matrix
used in the experiments. Candidate columns are zero-based tensor positions.
Class-weight sampling remains active in evaluation mode; `num_draws` controls
the number of classifier draws. A device-matched `torch.Generator` can be passed
to `forward` for reproducible sampling.

## Output tensors

For batch size `B`, class count `C`, latent width `D`, attribute width `A`, input
width `Dx`, and draw count `R`, the backbone returns the following tensors.

| Field | Shape |
| --- | --- |
| `features`, `sample_mean`, `sample_log_variance` | `[B, D]` |
| `semantic_features`, `class_mean`, `class_log_variance` | `[C, D]` |
| `classifier_samples` | `[R, C, D]` |
| `logits_samples` | `[R, B, C]` |
| `attribute_logits` | `[B, A]` |
| `reconstruction` | `[B, Dx]` |

`attribute_logits` are unnormalized linear-head outputs. `ModelConfig` contains
architecture settings only, not the complete experimental configuration.

## Files

```text
mgdsm/
  config.py           Architecture settings
  graph_utils.py      Attribute graph construction
  distributions.py    Gaussian weight sampling and probability averaging
  model.py            Encoders, auxiliary heads, and forward backbone
  __init__.py         Public component exports
demo_forward.py       Synthetic forward demonstration
tests/                Component behavior checks
```
