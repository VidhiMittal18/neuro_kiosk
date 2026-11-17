import io
import unittest
from unittest.mock import Mock, patch

import requests
import torch
import torch.nn as nn
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import edge_server as edge
import local_snn as LocalSNN


class ConstantLogitModel(nn.Module):
    def forward(self, tensor):
        return tensor.new_tensor([[2.0]])


def fake_snn():
    return LocalSNN.LoadedSNN(
        model=ConstantLogitModel(),
        threshold=0.8,
        timesteps=16,
        device=torch.device("cpu"),
    )


class UploadValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_sample_png(self):
        image_bytes = (edge.SAMPLES_DIR / "sample_normal.png").read_bytes()
        upload = UploadFile(
            file=io.BytesIO(image_bytes),
            filename="sample_normal.png",
            headers={"content-type": "image/png"},
        )

        received, image = await edge._read_upload(upload)

        self.assertEqual(received, image_bytes)
        self.assertEqual(image.mode, "L")

    async def test_rejects_mime_content_mismatch(self):
        image_bytes = (edge.SAMPLES_DIR / "sample_normal.png").read_bytes()
        upload = UploadFile(
            file=io.BytesIO(image_bytes),
            filename="sample_normal.jpg",
            headers={"content-type": "image/jpeg"},
        )

        with self.assertRaises(HTTPException) as raised:
            await edge._read_upload(upload)

        self.assertEqual(raised.exception.status_code, 422)

    async def test_rejects_oversized_file(self):
        upload = UploadFile(
            file=io.BytesIO(b"x" * (edge.MAX_IMAGE_BYTES + 1)),
            filename="large.png",
            headers={"content-type": "image/png"},
        )

        with self.assertRaises(HTTPException) as raised:
            await edge._read_upload(upload)

        self.assertEqual(raised.exception.status_code, 422)


class InferenceContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.model = fake_snn()
        edge.app.state.snn_model = self.model

    async def test_simulated_offline_returns_new_snn_scores(self):
        image_bytes = (edge.SAMPLES_DIR / "sample_normal.png").read_bytes()
        upload = UploadFile(
            file=io.BytesIO(image_bytes),
            filename="sample_normal.png",
            headers={"content-type": "image/png"},
        )

        response = await edge.diagnose(upload, simulate_offline=True)

        self.assertEqual(response["mode"], "offline")
        self.assertEqual(response["offline_reason"], "Simulated offline")
        self.assertEqual(response["result"]["time_steps"], 16)
        self.assertEqual(response["result"]["prediction"], "Pneumonia")
        self.assertNotIn("snn", response["result"])

    @patch("edge_server.requests.post")
    async def test_cloud_failure_falls_back_with_reason(self, post):
        post.side_effect = requests.exceptions.ConnectTimeout()
        image_bytes = (edge.SAMPLES_DIR / "sample_normal.png").read_bytes()
        upload = UploadFile(
            file=io.BytesIO(image_bytes),
            filename="sample_normal.png",
            headers={"content-type": "image/png"},
        )

        response = await edge.diagnose(upload, simulate_offline=False)

        self.assertEqual(response["mode"], "offline")
        self.assertIn("ConnectTimeout", response["offline_reason"])
        self.assertEqual(response["result"]["time_steps"], 16)

    @patch("edge_server.requests.post")
    async def test_cloud_success_uses_common_result_shape(self, post):
        cloud_response = Mock()
        cloud_response.json.return_value = {
            "prediction": "Normal",
            "confidence": 0.91,
            "pneumonia_score": 0.08,
            "threshold": 0.9981,
            "latency_ms": 4.2,
            "source": "Cloud-DenseNet121-ANN",
        }
        post.return_value = cloud_response
        image_bytes = (edge.SAMPLES_DIR / "sample_normal.png").read_bytes()
        upload = UploadFile(
            file=io.BytesIO(image_bytes),
            filename="sample_normal.png",
            headers={"content-type": "image/png"},
        )

        response = await edge.diagnose(upload, simulate_offline=False)

        self.assertEqual(response["mode"], "online")
        self.assertEqual(response["result"]["prediction"], "Normal")
        self.assertEqual(response["result"]["pneumonia_score"], 0.08)
        self.assertEqual(response["result"]["threshold"], 0.9981)
        self.assertEqual(response["result"]["source"], "Cloud-DenseNet121-ANN")
        self.assertIsNone(response["offline_reason"])
        self.assertNotIn("snn", response["result"])
        cloud_response.raise_for_status.assert_called_once()


class EdgeApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_loader = patch("edge_server.LocalSNN.load_model", return_value=fake_snn())
        cls.model_loader.start()
        cls.client_context = TestClient(edge.app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        cls.model_loader.stop()

    def test_status_reports_model_and_cloud(self):
        cloud_response = Mock(ok=True)
        cloud_response.json.return_value = {"model": "DenseNet121-ANN"}
        with patch("edge_server.requests.get", return_value=cloud_response):
            response = self.client.get("/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "cloud_reachable": True,
                "cloud_model": "DenseNet121-ANN",
                "snn_loaded": True,
            },
        )

    def test_request_body_limit_rejects_oversized_upload(self):
        response = self.client.post(
            "/diagnose",
            files={"file": ("large.png", b"x" * (edge.MAX_IMAGE_BYTES + 1), "image/png")},
            data={"simulate_offline": "true"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("5 MB", response.json()["detail"])

    def test_simulated_offline_diagnosis_over_http(self):
        image_bytes = (edge.SAMPLES_DIR / "sample_normal.png").read_bytes()
        response = self.client.post(
            "/diagnose",
            files={"file": ("sample_normal.png", image_bytes, "image/png")},
            data={"simulate_offline": "true"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "offline")
        self.assertEqual(response.json()["result"]["time_steps"], 16)
        self.assertNotIn("snn", response.json()["result"])

    def test_rejects_invalid_image_over_http(self):
        response = self.client.post(
            "/diagnose",
            files={"file": ("note.txt", b"not an image", "text/plain")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("PNG or JPEG", response.json()["detail"])

    def test_samples_are_whitelisted(self):
        allowed = self.client.get("/samples/sample_normal.png")
        rejected = self.client.get("/samples/unknown.png")

        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers["content-type"], "image/png")
        self.assertEqual(rejected.status_code, 404)


class StartupTests(unittest.TestCase):
    def test_snn_is_loaded_once_per_server_lifespan(self):
        with patch("edge_server.LocalSNN.load_model", return_value=fake_snn()) as load_model:
            with TestClient(edge.app):
                self.assertIsNotNone(edge.app.state.snn_model)

        load_model.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()