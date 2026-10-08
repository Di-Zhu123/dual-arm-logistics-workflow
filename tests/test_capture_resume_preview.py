import json
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np

from calibration.capture_dataset import LiveCaptureCoordinator, next_missing_index
from calibration.preview_three_cameras import PreviewHTTPServer, PreviewState


class CaptureResumeTests(unittest.TestCase):
    def test_failed_sample_number_is_reused(self):
        completed = {0, 1, 3}
        self.assertEqual(next_missing_index(completed, 5), 2)
        # A failed attempt does not mutate completed, so the same number is next.
        self.assertEqual(next_missing_index(completed, 5), 2)
        completed.add(2)
        self.assertEqual(next_missing_index(completed, 5), 4)
        completed.add(4)
        self.assertIsNone(next_missing_index(completed, 5))


class IntegratedPreviewTests(unittest.TestCase):
    def test_preview_and_snapshot_share_one_capture_producer(self):
        state = PreviewState()
        frame_numbers = {"left": 0, "head": 0, "right": 0}

        def fake_capture_camera(role, _settle_frames):
            time.sleep(0.005)
            frame_numbers[role] += 1
            started = time.time_ns()
            image = np.full((24, 32, 3), frame_numbers[role], dtype=np.uint8)
            depth = np.full((24, 32), 1000, dtype=np.uint16)
            metadata = {
                "host_capture_started_ns": started,
                "host_capture_ended_ns": time.time_ns(),
                "color_frame_number": frame_numbers[role],
            }
            return image, depth, metadata

        with patch(
            "calibration.capture_dataset.capture_camera",
            side_effect=fake_capture_camera,
        ):
            coordinator = LiveCaptureCoordinator(
                {"left": "left", "head": "head", "right": "right"}, state, 0
            )
            try:
                requested = time.time_ns()
                captures = coordinator.snapshot_after(requested, timeout_s=2.0)
                self.assertEqual(set(captures), {"left", "head", "right"})
                self.assertIsNotNone(state.get(None))
                self.assertTrue(all(state.get(role) for role in captures))
            finally:
                coordinator.close()

    def test_http_page_exposes_images_and_operator_status(self):
        state = PreviewState()
        images = {
            role: np.zeros((24, 32, 3), dtype=np.uint8)
            for role in ("left", "head", "right")
        }
        state.update(images, {role: 1 for role in images})
        state.set_operator_status("retry sample_004")
        server = PreviewHTTPServer(("127.0.0.1", 0), state)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            port = server.server_address[1]
            with urlopen(f"http://127.0.0.1:{port}/status.json", timeout=2) as response:
                status = json.loads(response.read().decode("utf-8"))
            self.assertEqual(status["operator_status"], "retry sample_004")
            with urlopen(
                f"http://127.0.0.1:{port}/snapshot/all.jpg", timeout=2
            ) as response:
                self.assertEqual(response.headers.get_content_type(), "image/jpeg")
                self.assertGreater(len(response.read()), 10)
        finally:
            state.stop.set()
            server.shutdown()
            server.server_close()
            worker.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
