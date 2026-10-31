"""Behavioral checks for the public model components."""
import math
import unittest

import numpy as np
import torch

from mgdsm import (
    MGDSMBackbone,
    ModelConfig,
    cosine_adjacency,
    mean_class_probabilities,
    normalize_adjacency,
    sample_class_weights,
)


class ModelComponentTests(unittest.TestCase):
    def setUp(self):
        attributes = np.random.default_rng(0).integers(0, 2, size=(6, 4))
        self.model = MGDSMBackbone(attributes, ModelConfig(input_dim=7, embed_dim=5))
        self.x = torch.randn(3, 7)

    def test_forward_supports_backpropagation(self):
        output = self.model(self.x, num_draws=2)
        self.assertEqual(output.logits_samples.shape, (2, 3, 6))
        self.assertEqual(output.sample_mean.shape, (3, 5))
        self.assertEqual(output.class_mean.shape, (6, 5))
        self.assertEqual(output.attribute_logits.shape, (3, 4))
        self.assertEqual(output.reconstruction.shape, self.x.shape)
        loss = sum(
            tensor.square().mean() for tensor in (
                output.logits_samples, output.reconstruction, output.attribute_logits,
                output.sample_mean, output.sample_log_variance,
            )
        )
        loss.backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_sample_gaussian_heads_do_not_change_classifier_inputs(self):
        def forward():
            return self.model(self.x, generator=torch.Generator().manual_seed(3))
        before = forward()
        with torch.no_grad():
            self.model.process_encoder.mean_head.bias.add_(10)
            self.model.process_encoder.logvar_head.bias.add_(10)
        after = forward()
        self.assertFalse(torch.equal(before.sample_mean, after.sample_mean))
        for name in ("features", "logits_samples", "reconstruction", "attribute_logits"):
            torch.testing.assert_close(getattr(before, name), getattr(after, name))

    def test_mc_aggregation_averages_probabilities(self):
        logits = torch.tensor([[[10., 0.]], [[-2., 0.]]])
        p = (1 / (1 + math.exp(-10)) + 1 / (1 + math.exp(2))) / 2
        expected = torch.tensor([[p, 1 - p]])
        torch.testing.assert_close(mean_class_probabilities(logits), expected)
        self.assertFalse(torch.allclose(expected, logits.mean(0).softmax(-1)))

    def test_graph_handles_isolated_and_single_nodes(self):
        for attributes in (np.ones((1, 4)), np.zeros((3, 4))):
            graph = cosine_adjacency(attributes, top_k=0)
            np.testing.assert_array_equal(graph, np.eye(len(attributes)))
            np.testing.assert_array_equal(normalize_adjacency(graph), graph)
        graph = normalize_adjacency(np.ones((2, 2)))
        np.testing.assert_allclose(graph, np.full((2, 2), 0.5), rtol=1e-6)

    def test_gaussian_log_variances_are_clipped(self):
        with torch.no_grad():
            self.model.process_encoder.logvar_head.bias.fill_(100)
            self.model.semantic_encoder.logvar_head.weight.fill_(100)
        output = self.model(self.x)
        for logvar in (output.sample_log_variance, output.class_log_variance):
            self.assertTrue((logvar <= self.model.config.logvar_max).all())
            self.assertTrue((logvar >= self.model.config.logvar_min).all())

    def test_state_roundtrip_preserves_graph_and_forward_outputs(self):
        restored = MGDSMBackbone(np.zeros((6, 4)), self.model.config)
        restored.load_state_dict(self.model.state_dict())
        one = self.model(self.x, generator=torch.Generator().manual_seed(2))
        two = restored(self.x, generator=torch.Generator().manual_seed(2))
        torch.testing.assert_close(self.model.normalized_adjacency, restored.normalized_adjacency)
        torch.testing.assert_close(one.logits_samples, two.logits_samples)
        torch.testing.assert_close(one.reconstruction, two.reconstruction)

    def test_invalid_shapes_and_sampling_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model(torch.randn(3, 8))
        with self.assertRaises(ValueError):
            self.model(self.x, num_draws=0)
        with self.assertRaises(ValueError):
            sample_class_weights(torch.ones(2, 3), torch.ones(3, 2))
        with self.assertRaises(ValueError):
            cosine_adjacency(np.ones((2, 4)), top_k=-1)
        with self.assertRaises(ValueError):
            normalize_adjacency(np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
