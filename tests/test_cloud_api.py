import io
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image
from torchvision import models

import cloud_api as Cloud
import team_model as Model


class ConstantLogitModel(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, tensor):
        return tensor.new_tensor([[self.value]])


def fake_ann(value=2.0, threshold=0.8):
    return Model.LoadedANN(
        model=ConstantLogitModel(value),
        threshold=threshold,
        device=torch.device("cpu"),
    )


class CloudModelTests(unittest.TestCase):
    def test_preprocessing_matches_dense_net_package_input(self):
        image = Image.new("RGB", (30, 20), color=(90, 90, 90))

        tensor = Model.PREPROCESS(image).unsqueeze(0)

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

    def test_ann_threshold_decides_class_and_returns_pneumonia_score(self):
        result = Model.predict_ann(fake_ann(), Image.new("L", (30, 20), color=90))

        self.assertEqual(result["prediction"], "Pneumonia")
        self.assertAlmostEqual(result["pneumonia_score"], torch.sigmoid(torch.tensor(2.0)).item())
        self.assertEqual(result["threshold"], 0.8)
        self.assertEqual(result["source"], "Cloud-DenseNet121-ANN")

    def test_ann_threshold_can_classify_normal(self):
        result = Model.predict_ann(
            fake_ann(value=-2.0),
            Image.new("L", (30, 20), color=90),
        )

        self.assertEqual(result["prediction"], "Normal")
        self.assertAlmostEqual(
            result["confidence"],
            1.0 - torch.sigmoid(torch.tensor(-2.0)).item(),
        )

    def test_bundled_ann_checkpoint_loads_strictly(self):
        loaded = Model.load_ann()

        self.assertIsInstance(loaded.model, models.DenseNet)
        self.assertEqual(loaded.model.classifier.out_features, 1)
        self.assertAlmostEqual(loaded.threshold, 0.9981153011322021)


class CloudAPITests(unittest.TestCase):
    def test_health_and_prediction_use_dense_net_ann(self):
        with patch("cloud_api.Model.load_ann", return_value=fake_ann()):
            with TestClient(Cloud.app) as client:
                health = client.get("/health")
                with Image.new("RGB", (30, 20), color=(90, 90, 90)) as image:
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                response = client.post(
                    "/predict",
                    files={"file": ("sample.png", buffer.getvalue(), "image/png")},
                )
                tensor_response = client.post(
                    "/predict",
                    data={"tensor_json": "[[90, 90], [90, 90]]"},
                )

        self.assertEqual(health.json(), {"status": "ok", "model": "DenseNet121-ANN"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "Pneumonia")
        self.assertEqual(response.json()["source"], "Cloud-DenseNet121-ANN")
        self.assertAlmostEqual(response.json()["threshold"], 0.8)
        self.assertEqual(tensor_response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
