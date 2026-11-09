import io
import unittest
from unittest.mock import Mock, patch

import requests
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import common as C
import edge_server as edge


class UploadValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_sample_png(self):
        image_bytes = (C.SAMPLES_DIR / "sample_normal.png").read_bytes()
        upload = UploadFile(
            file=io.BytesIO(image_bytes),
            filename="sample_normal.png",
            headers={"content-type": "image/png"},
        )

        received, image = await edge._read_upload(upload)

        self.assertEqual(received, image_bytes)
        self.assertEqual(image.mode, "L")

    async def test_rejects_mime_content_mismatch(self):
        image_bytes = (C.SAMPLES_DIR / "sample_normal.png").read_bytes()
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
        self.model = C.load_trained("snn", C.SNN_PATH, device="cpu")
        edge.app.state.snn_model = self.model

    async def test_simulated_offline_returns_snn_details(self):
        image_bytes = (C.SAMPLES_DIR / "sample_normal.png").read_bytes()
        upload = UploadFile(
            file=io.BytesIO(image_bytes),
            filename="sample_normal.png",
            headers={"content-type": "image/png"},
        )

        response = await edge.diagnose(upload, simulate_offline=True)

        self.assertEqual(response["mode"], "offline")
        self.assertEqual(response["offline_reason"], "Simulated offline")
        self.assertIn("snn", response["result"])
        self.assertIn("per_layer_spike_rate", response["result"]["snn"])

    @patch("edge_server.requests.post")
    async def test_cloud_failure_falls_back_with_reason(self, post):
        post.side_effect = requests.exceptions.ConnectTimeout()
        image_bytes = (C.SAMPLES_DIR / "sample_normal.png").read_bytes()
        upload = UploadFile(
            file=io.BytesIO(image_bytes),
            filename="sample_normal.png",
            headers={"content-type": "image/png"},
        )

        response = await edge.diagnose(upload, simulate_offline=False)

        self.assertEqual(response["mode"], "offline")
        self.assertIn("ConnectTimeout", response["offline_reason"])
        self.assertIn("snn", response["result"])

    @patch("edge_server.requests.post")
    async def test_cloud_success_uses_common_result_shape(self, post):
        cloud_response = Mock()
        cloud_response.json.return_value = {
            "prediction": "Normal",
            "confidence": 0.91,
            "latency_ms": 4.2,
            "source": "Cloud-API",
        }
        post.return_value = cloud_response
        image_bytes = (C.SAMPLES_DIR / "sample_normal.png").read_bytes()
        upload = UploadFile(
            file=io.BytesIO(image_bytes),
            filename="sample_normal.png",
            headers={"content-type": "image/png"},
        )

        response = await edge.diagnose(upload, simulate_offline=False)

        self.assertEqual(response["mode"], "online")
        self.assertEqual(response["result"]["prediction"], "Normal")
        self.assertIsNone(response["offline_reason"])
        self.assertNotIn("snn", response["result"])
        cloud_response.raise_for_status.assert_called_once()


class EdgeApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(edge.app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_status_reports_model_and_cloud(self):
        cloud_response = Mock(ok=True)
        with patch("edge_server.requests.get", return_value=cloud_response):
            response = self.client.get("/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"cloud_reachable": True, "snn_loaded": True})

    def test_benchmark_endpoints_serve_existing_assets(self):
        metrics = self.client.get("/benchmarks")
        figure = self.client.get("/benchmark-image")

        self.assertEqual(metrics.status_code, 200)
        self.assertEqual(metrics.json()["config"]["dataset"], "PneumoniaMNIST")
        self.assertEqual(figure.status_code, 200)
        self.assertEqual(figure.headers["content-type"], "image/png")

    def test_request_body_limit_rejects_oversized_upload(self):
        response = self.client.post(
            "/diagnose",
            files={"file": ("large.png", b"x" * (edge.MAX_IMAGE_BYTES + 1), "image/png")},
            data={"simulate_offline": "true"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("5 MB", response.json()["detail"])

    def test_simulated_offline_diagnosis_over_http(self):
        image_bytes = (C.SAMPLES_DIR / "sample_normal.png").read_bytes()
        response = self.client.post(
            "/diagnose",
            files={"file": ("sample_normal.png", image_bytes, "image/png")},
            data={"simulate_offline": "true"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "offline")
        self.assertIn("snn", response.json()["result"])

    def test_rejects_invalid_image_over_http(self):
        response = self.client.post(
            "/diagnose",
            files={"file": ("note.txt", b"not an image", "text/plain")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("PNG or JPEG", response.json()["detail"])

    def test_samples_are_whitelisted(self):
        allowed = self.client.get("/samples/sample_normal.png")
        rejected = self.client.get("/samples/../../trained_snn.pt")

        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers["content-type"], "image/png")
        self.assertEqual(rejected.status_code, 404)


class StartupTests(unittest.TestCase):
    def test_snn_is_loaded_once_per_server_lifespan(self):
        with patch("edge_server.C.load_trained", wraps=C.load_trained) as load_model:
            with TestClient(edge.app):
                self.assertIsNotNone(edge.app.state.snn_model)

        load_model.assert_called_once_with("snn", C.SNN_PATH, device="cpu")


if __name__ == "__main__":
    unittest.main()