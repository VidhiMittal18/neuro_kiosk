import unittest

import torch
import torch.nn as nn
from PIL import Image

import local_snn as LocalSNN


class ConstantLogitModel(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, tensor):
        self.calls += 1
        return tensor.new_tensor([[self.value]])


class LocalSNNTests(unittest.TestCase):
    def test_preprocessing_matches_package_shape_and_grayscale_channels(self):
        image = Image.new("RGB", (32, 24), color=(90, 90, 90))

        tensor = LocalSNN.PREPROCESS(image).unsqueeze(0)

        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))
        grayscale_value = 90 / 255
        expected = torch.tensor([
            (grayscale_value - mean) / std
            for mean, std in zip(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            )
        ]).view(1, 3, 1, 1)
        self.assertTrue(torch.allclose(tensor, expected.expand_as(tensor)))

    def test_predict_uses_metadata_threshold_and_all_timesteps(self):
        network = ConstantLogitModel(2.0)
        model = LocalSNN.LoadedSNN(
            model=network,
            threshold=0.8,
            timesteps=16,
            device=torch.device("cpu"),
        )

        result = LocalSNN.predict(model, Image.new("L", (40, 30), color=128))

        self.assertEqual(network.calls, 16)
        self.assertEqual(result["prediction"], "Pneumonia")
        self.assertEqual(result["time_steps"], 16)
        self.assertAlmostEqual(result["pneumonia_score"], torch.sigmoid(torch.tensor(2.0)).item())
        self.assertGreater(result["latency_ms"], 0)
        self.assertNotIn("snn", result)

    def test_predict_applies_threshold_for_normal_class(self):
        model = LocalSNN.LoadedSNN(
            model=ConstantLogitModel(-2.0),
            threshold=0.8,
            timesteps=4,
            device=torch.device("cpu"),
        )

        result = LocalSNN.predict(model, Image.new("L", (40, 30), color=128))

        self.assertEqual(result["prediction"], "Normal")
        self.assertAlmostEqual(result["confidence"], 1.0 - torch.sigmoid(torch.tensor(-2.0)).item())

    def test_bundled_checkpoint_loads_with_package_timestep_and_threshold(self):
        loaded = LocalSNN.load_model()

        self.assertEqual(loaded.timesteps, 16)
        self.assertAlmostEqual(loaded.threshold, 0.9999173879623413)
        self.assertIsInstance(loaded.model, nn.Module)


if __name__ == "__main__":
    unittest.main()
