"""Run the public neural components on synthetic inputs."""
from dataclasses import fields

import numpy as np
import torch

from mgdsm import MGDSMBackbone, mean_class_probabilities


def main() -> None:
    # This synthetic attribute matrix is not the experimental fault matrix.
    attributes = np.random.default_rng(0).integers(0, 2, size=(15, 20))
    model = MGDSMBackbone(attributes).eval()
    x = torch.randn(4, model.config.input_dim)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        output = model(x, num_draws=3, generator=generator)
        probabilities = mean_class_probabilities(output.logits_samples)
    for field in fields(output):
        print(f"{field.name:24s} {tuple(getattr(output, field.name).shape)}")
    print(f"{'mean_probabilities':24s} {tuple(probabilities.shape)}")
    print("Synthetic forward demonstration completed; no fault labels were predicted.")


if __name__ == "__main__":
    main()
