import unittest

from robot_workflow.observations import capture_required_wrists_optional_head


class _Camera:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    def get_framedata(self):
        if self.error is not None:
            raise self.error
        return dict(self.value)


class _Arm:
    def __init__(self, joint, camera):
        self.joint = joint
        self.camera = camera

    def get_joint_degree(self):
        return 0, [self.joint] * 7


class ObservationTests(unittest.TestCase):
    def test_head_timeout_does_not_discard_required_wrist_frames(self):
        environment = type(
            "Environment",
            (),
            {
                "arm_left": _Arm(1, _Camera({"rgb": "left", "depth": 1})),
                "arm_right": _Arm(2, _Camera({"rgb": "right", "depth": 2})),
                "head_camera": _Camera(error=RuntimeError("timeout")),
            },
        )()
        observation, head_error = capture_required_wrists_optional_head(environment)
        self.assertEqual(set(observation), {"left", "right"})
        self.assertIn("timeout", head_error)
        self.assertEqual(observation["left"]["joints"], [1] * 7)


if __name__ == "__main__":
    unittest.main()
