import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from robot_workflow.hardware_video import HardwareVideoRecorder


class _Camera:
    def __init__(self, value: int):
        self.value = value

    def get_framedata(self):
        time.sleep(0.002)
        return {
            "rgb": np.full((24, 32, 3), self.value, dtype=np.uint8),
            "depth": np.full((24, 32), self.value, dtype=np.uint16),
        }


class _FailingCamera:
    def get_framedata(self):
        time.sleep(0.002)
        raise RuntimeError("head timeout")


class _Writer:
    def __init__(self):
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(np.array(frame, copy=True))

    def release(self):
        self.released = True


class HardwareVideoRecorderTests(unittest.TestCase):
    def test_required_wrists_record_independently_of_failed_head(self):
        writers = {}

        def writer_factory(path, _fourcc, _fps, size):
            writer = _Writer()
            writers[path.name] = (writer, size)
            return writer

        with tempfile.TemporaryDirectory() as temporary:
            recorder = HardwareVideoRecorder(
                {"left": _Camera(1), "right": _Camera(2), "head": _FailingCamera()},
                Path(temporary),
                fps=30,
                writer_factory=writer_factory,
            )
            recorder.start()
            recorder.wait_for_required(timeout_s=1)
            requested = recorder.mark("final_approach")
            observation, errors = recorder.snapshot_after(requested, timeout_s=1)
            metadata = recorder.stop()

        self.assertEqual(set(observation), {"left", "right"})
        self.assertIn("head", errors)
        self.assertGreater(metadata["cameras"]["left"]["frame_count"], 0)
        self.assertGreater(metadata["cameras"]["right"]["frame_count"], 0)
        self.assertGreater(metadata["cameras"]["head"]["error_count"], 0)
        self.assertEqual(metadata["cameras"]["left"]["resolution"], [32, 24])
        self.assertTrue(writers["real_hardware_left.mp4"][0].released)
        self.assertTrue(writers["real_hardware_right.mp4"][0].released)
        phases = [marker["phase"] for marker in metadata["markers"]]
        self.assertIn("final_approach", phases)
        self.assertEqual(metadata["cameras"]["head"]["status"], "stopped")

    def test_required_video_must_start_before_motion(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = HardwareVideoRecorder(
                {"left": _FailingCamera(), "right": _Camera(2)},
                Path(temporary),
                fps=30,
                writer_factory=lambda *_args: _Writer(),
            )
            recorder.start()
            try:
                with self.assertRaisesRegex(RuntimeError, "left"):
                    recorder.wait_for_required(timeout_s=0.05)
            finally:
                recorder.stop()


if __name__ == "__main__":
    unittest.main()
