import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CalibrationSafetyContractTests(unittest.TestCase):
    def test_capture_source_has_no_robot_or_gripper_command(self):
        source = (ROOT / "calibration" / "capture_dataset.py").read_text("utf-8")
        forbidden = (
            ".action(",
            ".movej(",
            ".movejs(",
            ".movel(",
            ".open_gripper(",
            ".close_gripper(",
            ".continue_close_gripper(",
            "rm_movej(",
            "rm_movel(",
            "rm_set_gripper",
        )
        for call in forbidden:
            with self.subTest(call=call):
                self.assertNotIn(call, source)

    def test_capture_never_deletes_existing_data(self):
        source = (ROOT / "calibration" / "capture_dataset.py").read_text("utf-8")
        self.assertNotIn("rmtree", source)
        self.assertNotIn("unlink(", source)

    def test_handheld_capture_is_concurrent_and_requires_all_camera_corners(self):
        source = (ROOT / "calibration" / "capture_dataset.py").read_text("utf-8")
        self.assertIn("ThreadPoolExecutor(max_workers=3)", source)
        self.assertIn('"head_from_wrist_handheld"', source)
        self.assertIn('for name in ("left", "right")', source)

    def test_preview_has_no_robot_control_dependency(self):
        source = (ROOT / "calibration" / "preview_three_cameras.py").read_text("utf-8")
        self.assertNotIn("rm_robot_interface", source)
        self.assertNotIn(".action(", source)
        self.assertNotIn("movej", source.lower())
        self.assertIn("127.0.0.1", source)

    def test_example_camera_roles_are_unique_and_complete(self):
        document = json.loads(
            (ROOT / "config" / "camera_identity.example.json").read_text("utf-8")
        )
        cameras = document["cameras"]
        self.assertEqual(set(cameras), {"left", "head", "right"})
        serials = [str(cameras[role]["serial_number"]) for role in cameras]
        self.assertEqual(len(serials), len(set(serials)))
        self.assertTrue(all(serial.endswith("_CAMERA_SERIAL") for serial in serials))

    def test_identity_file_contains_no_host_or_login_secret(self):
        source = (ROOT / "config" / "camera_identity.example.json").read_text("utf-8")
        for forbidden in ("password", "secret", "124.16.", "192.168."):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
