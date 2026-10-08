import math
import unittest

import cv2
import numpy as np

from calibration.geometry import (
    euler_xyz_to_rotation,
    inverse,
    rotation_error_deg,
    transform,
    translation_error_m,
)
from calibration.solve_wrist_eye_in_hand_via_head import solve_transform_pair
from calibration.solve_wrist_fixed_board import _calibrate_method


class WristEyeInHandTests(unittest.TestCase):
    def test_recovers_end_from_camera_with_moving_board(self):
        expected_end_from_camera = transform(
            euler_xyz_to_rotation(math.radians(2), math.radians(-3), math.radians(91)),
            (0.091, -0.037, 0.044),
        )
        expected_world_from_head = transform(
            euler_xyz_to_rotation(math.radians(-121), math.radians(28), math.radians(-88)),
            (0.057, 0.029, 0.620),
        )
        world_from_ends = []
        wrist_from_heads = []
        for index in range(16):
            phase = 2.0 * math.pi * index / 16.0
            world_from_end = transform(
                euler_xyz_to_rotation(
                    math.radians(-40 + 55 * math.sin(phase)),
                    math.radians(-25 + 30 * math.cos(phase * 1.3)),
                    math.radians(-70 + 35 * math.sin(phase * 0.7)),
                ),
                (
                    0.22 + 0.06 * math.sin(phase),
                    0.08 + 0.07 * math.cos(phase),
                    0.47 + 0.05 * math.sin(phase * 1.7),
                ),
            )
            # From world_from_head @ head_from_wrist =
            #      world_from_end @ end_from_camera.
            head_from_wrist = (
                inverse(expected_world_from_head)
                @ world_from_end
                @ expected_end_from_camera
            )
            world_from_ends.append(world_from_end)
            wrist_from_heads.append(inverse(head_from_wrist))

        result = solve_transform_pair(world_from_ends, wrist_from_heads)
        self.assertLess(
            translation_error_m(result["end_from_camera"], expected_end_from_camera),
            1e-7,
        )
        self.assertLess(
            rotation_error_deg(result["end_from_camera"], expected_end_from_camera),
            1e-4,
        )
        self.assertLess(
            translation_error_m(result["world_from_head_camera"], expected_world_from_head),
            1e-7,
        )

    def test_rejects_static_arm_even_with_many_board_views(self):
        world_from_end = transform(np.eye(3), (0.2, 0.1, 0.5))
        world_from_ends = [world_from_end.copy() for _ in range(12)]
        wrist_from_heads = [
            transform(
                euler_xyz_to_rotation(0.01 * index, -0.02 * index, 0.005 * index),
                (0.3 + 0.01 * index, 0.0, -0.1),
            )
            for index in range(12)
        ]
        with self.assertRaisesRegex(ValueError, "insufficient arm motion"):
            solve_transform_pair(world_from_ends, wrist_from_heads)

    def test_fixed_board_standard_handeye_recovers_end_from_camera(self):
        expected_end_from_camera = transform(
            euler_xyz_to_rotation(math.radians(3), math.radians(-4), math.radians(89)),
            (0.083, -0.041, 0.043),
        )
        world_from_board = transform(
            euler_xyz_to_rotation(math.radians(-7), math.radians(11), math.radians(4)),
            (0.31, -0.08, 0.73),
        )
        world_from_ends = []
        camera_from_boards = []
        for index in range(12):
            phase = 2.0 * math.pi * index / 12.0
            world_from_end = transform(
                euler_xyz_to_rotation(
                    math.radians(-32 + 48 * math.sin(phase)),
                    math.radians(18 + 35 * math.cos(phase * 1.2)),
                    math.radians(-64 + 31 * math.sin(phase * 0.8)),
                ),
                (0.2 + 0.05 * math.sin(phase), 0.1 + 0.04 * math.cos(phase), 0.5),
            )
            camera_from_board = inverse(world_from_end @ expected_end_from_camera) @ world_from_board
            world_from_ends.append(world_from_end)
            camera_from_boards.append(camera_from_board)

        recovered = _calibrate_method(
            world_from_ends,
            camera_from_boards,
            cv2.CALIB_HAND_EYE_PARK,
        )
        self.assertLess(translation_error_m(recovered, expected_end_from_camera), 1e-7)
        self.assertLess(rotation_error_deg(recovered, expected_end_from_camera), 1e-4)


if __name__ == "__main__":
    unittest.main()
