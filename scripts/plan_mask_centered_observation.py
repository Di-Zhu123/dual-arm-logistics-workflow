#!/usr/bin/env python3
"""Plan a near observation directly above a SAM-mask point-cloud center."""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from robot_workflow.legacy_tcp import LegacyApiGateway


# Repeated hardware captures found the best usable near view at z=0.35395 m.
# Plan to that verified height in short, chained simulation segments so the
# solver stays on the reachable branch without requiring camera recaptures.
FAST_NEAR_VIEW_CAMERA_Z_M = 0.354
FAST_NEAR_VIEW_STEP_M = 0.020
FAST_NEAR_VIEW_TOLERANCE_M = 0.002
FAST_NEAR_VIEW_MAX_STAGES = 5
FAST_NEAR_VIEW_WAYPOINTS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_plan", type=Path)
    parser.add_argument("--observation-z", type=float, default=0.40)
    parser.add_argument(
        "--robotdata",
        type=Path,
        default=Path("RobotDataCollection"),
    )
    return parser.parse_args()


def camera_pose_to_arm_end_pose(
    camera_pose: list[float],
    camera_extrinsic: dict[str, Any],
    rotation_from_euler: Any,
    euler_from_rotation: Any,
) -> list[float]:
    """Convert world-from-camera to world-from-Link7 without solving IK."""

    world_from_camera_rotation = rotation_from_euler(*camera_pose[3:])
    world_from_camera_translation = np.asarray(camera_pose[:3], dtype=float)
    body_from_camera_rotation = np.asarray(camera_extrinsic["R"], dtype=float)
    body_from_camera_translation = np.asarray(camera_extrinsic["t"], dtype=float)
    world_from_body_rotation = (
        world_from_camera_rotation @ body_from_camera_rotation.T
    )
    world_from_body_translation = (
        world_from_camera_translation
        - world_from_body_rotation @ body_from_camera_translation
    )
    return [
        *world_from_body_translation.tolist(),
        *euler_from_rotation(world_from_body_rotation),
    ]


def main() -> int:
    arguments = parse_args()
    if not 0.27 <= arguments.observation_z <= 0.40:
        raise ValueError("observation-z must be within 0.27..0.40 m")
    capture = json.loads(arguments.capture_plan.read_text("utf-8"))
    center = [float(value) for value in capture["sam_mask_world_center_m"]]
    arm_name = capture["arm"]
    current_joints = [float(value) for value in capture["current_joints"]]

    sys.path.insert(0, str(arguments.robotdata))
    from utils.kinematic_transforms import (  # type: ignore
        CameraKinematics,
        euler_angles_from_rotation_matrix,
        rotation_matrix_from_euler_angles,
    )

    configuration = json.loads(
        (arguments.robotdata / "config" / "env_config.json").read_text("utf-8")
    )[f"{arm_name}_arm_config"]
    camera = CameraKinematics(
        configuration["base_extrinsic"]["R"],
        configuration["base_extrinsic"]["t"],
        configuration["cam_extrinsic"]["R"],
        configuration["cam_extrinsic"]["t"],
    )
    fallback_heights = (0.38, 0.36, 0.34, 0.32, 0.30)
    candidate_heights = [float(arguments.observation_z)]
    candidate_heights.extend(
        value
        for value in fallback_heights
        if value < arguments.observation_z - 1e-9
    )
    gateway = LegacyApiGateway("127.0.0.1", timeout_s=120.0)
    attempts: list[dict[str, Any]] = []
    selected = None
    for height_m in candidate_heights:
        target_camera_pose = [
            center[0], center[1], height_m, math.pi, 0.0, -math.pi / 2
        ]
        target_joints = camera.get_camera_inverse(current_joints, target_camera_pose)
        if target_joints is not None:
            target_joints = [float(value) for value in target_joints]
        target_end_pose = camera_pose_to_arm_end_pose(
            target_camera_pose,
            configuration["cam_extrinsic"],
            rotation_matrix_from_euler_angles,
            euler_angles_from_rotation_matrix,
        )
        simulation = gateway.simulate(
            target_pose=target_end_pose,
            arm=arm_name,
            left_joints_deg=capture["left_joints"],
            right_joints_deg=capture["right_joints"],
            num_waypoints=25,
            recording=True,
        )
        attempt = {
            "target_height_m": height_m,
            "target_camera_pose": target_camera_pose,
            "local_ik_solved": target_joints is not None,
            "target_pose_source": "camera pose plus fixed camera extrinsic",
            "target_joints": target_joints,
            "target_end_pose": target_end_pose,
            "simulation_solved": bool(simulation["success"]),
        }
        attempts.append(attempt)
        if simulation["success"]:
            attempt["waypoints"] = simulation["path"]
            selected = (height_m, target_camera_pose, target_joints, target_end_pose, simulation)
            break
    if selected is None:
        raise RuntimeError("simulation could not plan any mask-centered observation height")
    selected_height_m, target_camera_pose, target_joints, target_end_pose, simulation = selected

    selected_attempt = next(
        attempt for attempt in attempts if attempt.get("simulation_solved")
    )
    combined_waypoints = [
        [float(value) for value in waypoint] for waypoint in simulation["path"]
    ]
    selected_attempt["initial_target_camera_pose"] = list(target_camera_pose)
    selected_attempt["initial_waypoints"] = list(combined_waypoints)
    segment_joints = list(combined_waypoints[-1])
    fast_target_z_m = min(
        float(arguments.observation_z), FAST_NEAR_VIEW_CAMERA_Z_M
    )
    fast_near_view_stages: list[dict[str, Any]] = []
    for stage_index in range(1, FAST_NEAR_VIEW_MAX_STAGES + 1):
        achieved_camera_pose = [
            float(value) for value in camera.get_camera_forward(segment_joints)
        ]
        if (
            achieved_camera_pose[2]
            >= fast_target_z_m - FAST_NEAR_VIEW_TOLERANCE_M
        ):
            break
        stage_target_camera_pose = list(achieved_camera_pose)
        stage_target_camera_pose[2] = min(
            fast_target_z_m,
            achieved_camera_pose[2] + FAST_NEAR_VIEW_STEP_M,
        )
        stage_target_end_pose = camera_pose_to_arm_end_pose(
            stage_target_camera_pose,
            configuration["cam_extrinsic"],
            rotation_matrix_from_euler_angles,
            euler_angles_from_rotation_matrix,
        )
        stage_left_joints = (
            segment_joints if arm_name == "left" else capture["left_joints"]
        )
        stage_right_joints = (
            segment_joints if arm_name == "right" else capture["right_joints"]
        )
        stage_simulation = gateway.simulate(
            target_pose=stage_target_end_pose,
            arm=arm_name,
            left_joints_deg=stage_left_joints,
            right_joints_deg=stage_right_joints,
            num_waypoints=FAST_NEAR_VIEW_WAYPOINTS,
            recording=False,
        )
        stage_record = {
            "stage": stage_index,
            "start_camera_pose": achieved_camera_pose,
            "target_camera_pose": stage_target_camera_pose,
            "target_end_pose": stage_target_end_pose,
            "simulation_solved": bool(stage_simulation["success"]),
        }
        fast_near_view_stages.append(stage_record)
        if not stage_simulation["success"]:
            break
        stage_path = [
            [float(value) for value in waypoint]
            for waypoint in stage_simulation["path"]
        ]
        if not stage_path:
            break
        stage_record["waypoint_count"] = len(stage_path)
        combined_waypoints.extend(stage_path[1:])
        segment_joints = list(stage_path[-1])

    achieved_camera_pose = [
        float(value) for value in camera.get_camera_forward(segment_joints)
    ]
    selected_attempt["waypoints"] = combined_waypoints
    selected_attempt["target_camera_pose"] = achieved_camera_pose
    selected_attempt["fast_near_view_stages"] = fast_near_view_stages
    selected_attempt["fast_near_view_target_z_m"] = fast_target_z_m
    selected_attempt["fast_near_view_achieved_z_m"] = achieved_camera_pose[2]
    selected_height_m = achieved_camera_pose[2]
    target_camera_pose = achieved_camera_pose
    target_joints = segment_joints

    output = arguments.capture_plan.parent
    video_path = output / "mask_centered_observation_preview.mp4"
    video_path.write_bytes(base64.b64decode(simulation["video"]))
    adjustment: dict[str, Any] = {
        "kind": "mask_point_cloud_center",
        "reason": "move wrist camera as high as reachable above SAM-mask world center",
    }
    plan: dict[str, Any] = {
        "ok": True,
        "planning_only": True,
        "real_robot_command_sent": False,
        "gripper_command_sent": False,
        "status": "awaiting_human_review",
        "arm": arm_name,
        "current_joints": current_joints,
        "left_joints": capture["left_joints"],
        "right_joints": capture["right_joints"],
        "sam_mask_world_center_m": center,
        "requested_observation_height_m": float(arguments.observation_z),
        "fast_near_view_target_height_m": fast_target_z_m,
        "fast_near_view_achieved_height_m": achieved_camera_pose[2],
        "fast_near_view_stages": fast_near_view_stages,
        "selected_observation_height_m": float(selected_height_m),
        "current_camera_pose": capture["current_camera_pose"],
        "selected_adjustment": adjustment,
        "selected_target_joints": target_joints,
        "selected_target_end_pose": target_end_pose,
        "preview_video": str(video_path),
        "attempts": attempts,
    }
    plan_path = output / "mask_centered_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps({"plan": str(plan_path), **plan}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
