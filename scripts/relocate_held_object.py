#!/usr/bin/env python3
"""Relocate the currently held inert prop a short distance and release it."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

import numpy as np

from robot_workflow.legacy_tcp import LegacyApiGateway


JOINT_LOWER_DEG = np.array([-90.0, -105.0, -90.0, -165.0, -90.0, -97.8, -172.0])
JOINT_UPPER_DEG = np.array([90.0, 105.0, 90.0, 55.0, 90.0, 102.2, 172.0])
MINIMUM_JOINT_MARGIN_DEG = 3.0
LIFT_M = 0.05
LOWER_M = 0.025
TRANSLATE_Y_M = 0.05
SEGMENT_WAYPOINTS = 5
VERTICAL_SEGMENTS = 5
LOWER_SEGMENTS = 3
TRANSLATE_SEGMENTS = 5
MAX_SEGMENT_POSITION_ERROR_MM = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robotdata",
        type=Path,
        default=Path("RobotDataCollection"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("grasp_api_tests"))
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--execute-confirmed-held-inert-prop", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def simulate_segment(
    gateway: LegacyApiGateway,
    kinematics: Any,
    left_joints: Sequence[float],
    right_joints: Sequence[float],
    target_gripper_pose: Sequence[float],
    name: str,
) -> dict[str, Any]:
    response = gateway.simulate(
        target_pose=kinematics.gripper_to_end(target_gripper_pose),
        arm="left",
        left_joints_deg=left_joints,
        right_joints_deg=right_joints,
        num_waypoints=SEGMENT_WAYPOINTS,
        recording=True,
    )
    require(bool(response["success"]), f"simulation failed for {name}")
    path = [[float(value) for value in waypoint] for waypoint in response["path"]]
    require(2 <= len(path) <= SEGMENT_WAYPOINTS, f"invalid path for {name}")
    start_error = float(
        np.max(np.abs(np.asarray(path[0]) - np.asarray(left_joints)))
    )
    require(start_error <= 0.1, f"stale simulation start for {name}")
    actual_target = np.asarray(kinematics.get_gripper_forward(path[-1]), dtype=float)
    target = np.asarray(target_gripper_pose, dtype=float)
    position_error_mm = float(np.linalg.norm(actual_target[:3] - target[:3]) * 1000.0)
    require(
        position_error_mm <= MAX_SEGMENT_POSITION_ERROR_MM,
        f"FK endpoint mismatch for {name}: {position_error_mm:.3f} mm",
    )
    return {
        "name": name,
        "path": path,
        "start_error_deg": start_error,
        "position_error_mm": position_error_mm,
    }


def main() -> int:
    arguments = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = arguments.output_root / f"relocate_held_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(arguments.robotdata))
    from utils.arm_environment import DualArmEnvironment  # type: ignore

    config = json.loads(
        (arguments.robotdata / "config/env_config.json").read_text("utf-8")
    )
    environment = None
    result: dict[str, Any] = {
        "ok": False,
        "planning_only": not arguments.execute_confirmed_held_inert_prop,
        "real_robot_motion_sent": False,
        "gripper_release_sent": False,
        "lift_m": LIFT_M,
        "lower_m": LOWER_M,
        "translate_y_m": TRANSLATE_Y_M,
    }
    try:
        environment = DualArmEnvironment(
            config["left_arm_config"],
            config["right_arm_config"],
            config["head_camera_config"],
        )
        left_status, left_joints = environment.arm_left.get_joint_degree()
        right_status, right_joints = environment.arm_right.get_joint_degree()
        require(left_status == 0 and right_status == 0, "cannot read current joints")
        left_joints = [float(value) for value in left_joints]
        right_joints = [float(value) for value in right_joints]
        kinematics = environment.arm_left.gripper_kinematics
        start_pose = [
            float(value) for value in kinematics.get_gripper_forward(left_joints)
        ]
        gateway = LegacyApiGateway(arguments.api_host, timeout_s=120.0)

        segments = []
        current_joints = left_joints
        current_pose = list(start_pose)
        for index in range(1, VERTICAL_SEGMENTS + 1):
            target = list(current_pose)
            target[2] += LIFT_M / VERTICAL_SEGMENTS
            segment = simulate_segment(
                gateway, kinematics, current_joints, right_joints, target, f"lift_{index}"
            )
            segments.append(segment)
            current_joints = segment["path"][-1]
            current_pose = [
                float(value) for value in kinematics.get_gripper_forward(current_joints)
            ]
        for index in range(1, TRANSLATE_SEGMENTS + 1):
            target = list(current_pose)
            target[1] += TRANSLATE_Y_M / TRANSLATE_SEGMENTS
            segment = simulate_segment(
                gateway,
                kinematics,
                current_joints,
                right_joints,
                target,
                f"translate_{index}",
            )
            segments.append(segment)
            current_joints = segment["path"][-1]
            current_pose = [
                float(value) for value in kinematics.get_gripper_forward(current_joints)
            ]
        for index in range(1, LOWER_SEGMENTS + 1):
            target = list(current_pose)
            target[2] -= LOWER_M / LOWER_SEGMENTS
            segment = simulate_segment(
                gateway, kinematics, current_joints, right_joints, target, f"lower_{index}"
            )
            segments.append(segment)
            current_joints = segment["path"][-1]
            current_pose = [
                float(value) for value in kinematics.get_gripper_forward(current_joints)
            ]

        combined_path: list[list[float]] = []
        for segment in segments:
            combined_path.extend(
                segment["path"] if not combined_path else segment["path"][1:]
            )
        path_array = np.asarray(combined_path, dtype=float)
        margins = np.minimum(
            path_array.min(axis=0) - JOINT_LOWER_DEG,
            JOINT_UPPER_DEG - path_array.max(axis=0),
        )
        smallest_margin = float(margins.min())
        require(
            smallest_margin >= MINIMUM_JOINT_MARGIN_DEG,
            "relocation path has insufficient joint margin",
        )
        poses = np.asarray(
            [kinematics.get_gripper_forward(waypoint) for waypoint in combined_path],
            dtype=float,
        )
        achieved_lift_m = float(poses[:, 2].max() - start_pose[2])
        release_height_delta_m = float(poses[-1, 2] - start_pose[2])
        require(achieved_lift_m >= 0.03, "relocation does not achieve 30 mm lift")
        require(
            release_height_delta_m >= 0.005,
            "relocation release pose is not at least 5 mm above the start pose",
        )
        result.update(
            {
                "start_joints": left_joints,
                "right_joints": right_joints,
                "start_gripper_pose": start_pose,
                "target_gripper_pose": poses[-1].tolist(),
                "segments": segments,
                "combined_waypoint_count": len(combined_path),
                "smallest_joint_margin_deg": smallest_margin,
                "maximum_gripper_z_m": float(poses[:, 2].max()),
                "minimum_gripper_z_m": float(poses[:, 2].min()),
                "achieved_lift_m": achieved_lift_m,
                "release_height_delta_m": release_height_delta_m,
            }
        )
        (output / "plan.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not arguments.execute_confirmed_held_inert_prop:
            result["ok"] = True
            result["status"] = "planned_only"
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 0

        execution = []
        for index, waypoint in enumerate(combined_path[1:], start=1):
            return_code = environment.arm_left.robot.rm_movej(
                waypoint, v=2, r=0, connect=0, block=1
            )
            result["real_robot_motion_sent"] = True
            query_return, actual = environment.arm_left.get_joint_degree()
            error = float(np.max(np.abs(np.asarray(actual) - np.asarray(waypoint))))
            execution.append(
                {"index": index, "return": return_code, "error_deg": error}
            )
            result["execution"] = execution
            require(
                return_code == 0 and query_return == 0 and error <= 1.0,
                f"relocation waypoint {index} failed",
            )
        release_return = environment.arm_left.robot.rm_set_gripper_release(
            200, block=True, timeout=5
        )
        result["gripper_release_sent"] = True
        result["gripper_release_return"] = release_return
        require(release_return == 0, "gripper release failed at relocation target")
        result["ok"] = True
        result["status"] = "relocated_50mm_positive_y_and_released"
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        if environment is not None and result["real_robot_motion_sent"]:
            try:
                result["emergency_stop_return"] = environment.arm_left.robot.rm_set_arm_stop()
            except Exception as stop_error:
                result["emergency_stop_error"] = repr(stop_error)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1
    finally:
        (output / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if environment is not None:
            if hasattr(environment, "head_camera"):
                environment.head_camera.close()
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
