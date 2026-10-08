#!/usr/bin/env python3
"""Plan, simulate and optionally execute the original dual-wrist view stages.

The two supported stages reproduce the old workflow without importing or
modifying it: ``initial`` sends both wrists to the fixed search posture, while
``look-at`` aims both wrist cameras at a previously measured SAM point-cloud
center.  The initial stage can explicitly release the left gripper before any
reset motion so a previously grasped object is never carried through reset.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

from robot_workflow.legacy_tcp import LegacyApiGateway
from robot_workflow.observations import capture_required_wrists_optional_head


INITIAL_VIEW_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, -70.0, 0.0]
LOOK_IK_SEED = [0.0, 0.0, 0.0, -90.0, 0.0, 0.0, 0.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initial", "look-at"))
    parser.add_argument("--center-plan", type=Path)
    parser.add_argument(
        "--robotdata",
        type=Path,
        default=Path("RobotDataCollection"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("grasp_api_tests"))
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--execute-approved-observation", action="store_true")
    parser.add_argument("--release-left-before-initial", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def look_rotation(position: np.ndarray, eye: np.ndarray) -> np.ndarray:
    z_axis = position - eye
    z_axis /= np.linalg.norm(z_axis)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z_axis, up))) > 1.0 - 1e-6:
        up = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(z_axis, up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def euler_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    ry = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(ry)) > 1e-6:
        rz = math.atan2(matrix[1, 0], matrix[0, 0])
        rx = math.atan2(matrix[2, 1], matrix[2, 2])
    else:
        rz = 0.0
        rx = math.atan2(matrix[0, 1], matrix[1, 1])
    return rx, ry, rz


def read_joints(environment: Any) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for name, arm in (("left", environment.arm_left), ("right", environment.arm_right)):
        status, joints = arm.get_joint_degree()
        require(status == 0 and len(joints) == 7, f"failed to read {name} arm joints")
        result[name] = [float(value) for value in joints]
    return result


def target_joints(stage: str, arm: Any, center: list[float] | None) -> list[float]:
    if stage == "initial":
        return list(INITIAL_VIEW_JOINTS)
    require(center is not None and len(center) == 3, "look-at requires a 3D center")
    base = np.asarray(arm.base_coordinate, dtype=float)
    eye = base + np.array([0.2, 0.0, 0.5])
    rotation = look_rotation(np.asarray(center, dtype=float), eye)
    pose = [*eye.tolist(), *euler_from_matrix(rotation)]
    solved = arm.camera_kinematics.get_camera_inverse(LOOK_IK_SEED, pose)
    require(solved is not None, "camera IK could not solve look-at pose")
    return [float(value) for value in solved]


def save_frames(observation: dict[str, Any], output: Path) -> None:
    for name in ("left", "head", "right"):
        if name not in observation:
            continue
        rgb = np.asarray(observation[name]["rgb"])
        depth = np.asarray(observation[name]["depth"])
        cv2.imwrite(str(output / f"{name}_rgb.jpg"), rgb[:, :, ::-1])
        np.save(output / f"{name}_depth.npy", depth, allow_pickle=False)


def main() -> int:
    arguments = parse_args()
    require(
        arguments.stage == "initial" or arguments.center_plan is not None,
        "look-at requires --center-plan",
    )
    center = None
    if arguments.center_plan is not None:
        center_plan = json.loads(arguments.center_plan.read_text("utf-8"))
        center = [float(value) for value in center_plan["sam_mask_world_center_m"]]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = arguments.output_root / f"dual_wrist_{arguments.stage}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(arguments.robotdata))
    from utils.arm_environment import DualArmEnvironment  # type: ignore

    config = json.loads(
        (arguments.robotdata / "config/env_config.json").read_text("utf-8")
    )
    environment = None
    result: dict[str, Any] = {
        "ok": False,
        "stage": arguments.stage,
        "planning_only": not arguments.execute_approved_observation,
        "real_robot_command_sent": False,
        "gripper_command_sent": False,
        "sam_mask_world_center_m": center,
        "arms": [],
    }
    try:
        environment = DualArmEnvironment(
            config["left_arm_config"],
            config["right_arm_config"],
            config["head_camera_config"],
        )
        gateway = LegacyApiGateway(arguments.api_host, timeout_s=120.0)
        if arguments.release_left_before_initial:
            require(arguments.stage == "initial", "gripper release is only valid for initial")
            require(
                arguments.execute_approved_observation,
                "gripper release requires approved observation execution",
            )
            release_return = environment.arm_left.robot.rm_set_gripper_release(
                200, block=True, timeout=5
            )
            require(release_return == 0, f"pre-reset gripper release failed: {release_return}")
            result["gripper_command_sent"] = True
            result["pre_reset_left_gripper_release"] = {
                "sent": True,
                "return": release_return,
                "completed_before_arm_motion": True,
            }
        for name, arm in (("left", environment.arm_left), ("right", environment.arm_right)):
            joints_before = read_joints(environment)
            target = target_joints(arguments.stage, arm, center)
            target_end_pose = [
                float(value) for value in arm.gripper_kinematics.get_arm_end_forward(target)
            ]
            simulation = gateway.simulate(
                target_pose=target_end_pose,
                target_joints_deg=target,
                arm=name,
                left_joints_deg=joints_before["left"],
                right_joints_deg=joints_before["right"],
                num_waypoints=25,
                recording=True,
            )
            require(bool(simulation["success"]), f"simulation failed for {name} arm")
            waypoints = [[float(value) for value in point] for point in simulation["path"]]
            require(1 <= len(waypoints) <= 25, "invalid simulated waypoint count")
            require(
                all(
                    max(abs(a - b) for a, b in zip(previous, current)) <= 15.0
                    for previous, current in zip(waypoints, waypoints[1:])
                ),
                f"{name} simulated waypoint jump exceeds 15 degrees",
            )
            preview = output / f"{name}_preview.mp4"
            preview.write_bytes(base64.b64decode(simulation["video"]))
            arm_result = {
                "arm": name,
                "joints_before": joints_before[name],
                "target_joints": target,
                "target_end_pose": target_end_pose,
                "waypoints": waypoints,
                "preview_video": str(preview),
                "executed": False,
            }
            result["arms"].append(arm_result)
            if arguments.execute_approved_observation:
                require(
                    max(abs(a - b) for a, b in zip(joints_before[name], waypoints[0])) <= 0.75,
                    f"{name} simulation starts from a stale state",
                )
                for index, waypoint in enumerate(waypoints):
                    require(arm.movej(waypoint) == 0, f"{name} rejected waypoint {index}")
                result["real_robot_command_sent"] = True
                status, final = arm.get_joint_degree()
                require(status == 0 and len(final) == 7, f"failed to verify {name} endpoint")
                final = [float(value) for value in final]
                require(
                    max(abs(a - b) for a, b in zip(final, waypoints[-1])) <= 1.0,
                    f"{name} endpoint joint error exceeds one degree",
                )
                arm_result["executed"] = True
                arm_result["joints_after"] = final

        final_joints = read_joints(environment)
        frame_archive_errors = []
        for _ in range(2):
            try:
                observation, head_capture_error = (
                    capture_required_wrists_optional_head(environment)
                )
                save_frames(observation, output)
                if head_capture_error is not None:
                    frame_archive_errors.append(head_capture_error)
                break
            except RuntimeError as error:
                frame_archive_errors.append(repr(error))
        if frame_archive_errors:
            result["frame_archive_errors"] = frame_archive_errors
            result["frames_archived"] = len(frame_archive_errors) < 2
        else:
            result["frames_archived"] = True
        result.update(
            {
                "ok": True,
                "left_joints": final_joints["left"],
                "right_joints": final_joints["right"],
                "arm": "left",
                "current_joints": final_joints["left"],
                "current_camera_pose": [
                    float(value)
                    for value in environment.arm_left.camera_kinematics.get_camera_forward(
                        final_joints["left"]
                    )
                ],
            }
        )
        state_path = output / "state.json"
        state_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
        print(json.dumps({"output": str(output), "state": str(state_path), **result}, ensure_ascii=False))
        return 0
    finally:
        if environment is not None:
            if hasattr(environment, "head_camera"):
                environment.head_camera.close()
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
