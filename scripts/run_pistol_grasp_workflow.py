#!/usr/bin/env python3
"""Run the complete pistol observation and grasp workflow without LLM decisions.

This is an explicit, single-run hardware entry point.  It reuses the planning-only
scripts and existing APIs, applies deterministic geometric/kinematic filters, and
after closing the gripper executes a separately simulated, orientation-preserving
vertical lift, transfers the held object above the configured green box, releases
it inside the opening, and retreats vertically from the box.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

import cv2
import numpy as np

from robot_workflow.gripper_clearance import (
    MINIMUM_TABLE_CLEARANCE_M,
    VIRTUAL_COLLISION_LENGTH_M,
    virtual_tip_clearance_m,
)
from robot_workflow.hardware_video import HardwareVideoRecorder
from robot_workflow.legacy_tcp import LegacyApiGateway
from robot_workflow.observations import capture_required_wrists_optional_head


MAX_GRIPPER_WIDTH_M = 0.0654
# RealMan reports gripper aperture on an approximately 0..1000 scale.  Allow
# five millimetres of compression/geometry error relative to the depth-derived
# target cross-section, but do not treat a nearly closed gripper as a held item.
GRIPPER_OPEN_ACTPOS = 1000.0
HOLD_APERTURE_TOLERANCE_M = 0.005
MINIMUM_WRIST_TARGET_DEPTH_PIXELS = 1000
# A rigidly held target remains at nearly constant range in the wrist camera.
# Allow perspective/gravity motion, but not the roughly 65--100 mm range jump
# seen when the camera rises while the pistol stays on the table.
MAXIMUM_WRIST_NEAR_DEPTH_INCREASE_MM = 40.0
STRICT_CROSS_SECTION_WIDTH_M = (0.015, 0.058)
RELAXED_CROSS_SECTION_WIDTH_M = (0.012, 0.060)
STRICT_MINIMUM_SIDE_CLEARANCE_M = 0.003
# In relaxed mode, allow a small one-sided overlap so the open finger may push a
# lightweight target toward the grasp centre instead of rejecting the pose.
RELAXED_MINIMUM_SIDE_CLEARANCE_M = -0.005
MAGAZINE_CROSS_SECTION_WIDTH_M = (0.008, 0.063)
MAGAZINE_MINIMUM_SIDE_CLEARANCE_M = -0.008
MAGAZINE_MINIMUM_CROSS_SECTION_POINTS = 80
# The preferred tier matches the relaxed hard floor: once physical fit is
# established, distance to the left-arm base chooses the near-side grasp.
# Negative side clearance remains available as a no-stop fallback, but a pose
# with both fingers geometrically clear must win the preferred tier.  Hardware
# trials showed that the former -5 mm preferred threshold could stop the gripper
# on the side of the prop while it was still about 51 mm open.
PREFERRED_MINIMUM_SIDE_CLEARANCE_M = 0.0
MINIMUM_JOINT_MARGIN_DEG = 3.0
REQUIRED_PREGRASP_RETRACTION_M = 0.05
MAX_APPROACH_LATERAL_DEVIATION_M = 0.005
MAX_WRIST7_TO_PREGRASP_DELTA_DEG = 90.0
MAGAZINE_MAX_WRIST7_TO_PREGRASP_DELTA_DEG = 140.0
POST_GRASP_LIFT_M = 0.10
POST_GRASP_LIFT_SEGMENTS = 4
POST_GRASP_LIFT_WAYPOINTS_PER_SEGMENT = 5
MAX_LIFT_LATERAL_DEVIATION_M = 0.005
# The green box is kept at a fixed, measured workstation pose.  The near/front
# left rim was observed at (0.63, -0.135, 0.058) m; the padded opening is about
# 0.34 x 0.14 m, so this point stays well inside all four walls.
BOX_DROP_X_M = 0.45
BOX_DROP_Y_M = -0.21
BOX_RIM_Z_M = 0.058
BOX_OPENING_LENGTH_M = 0.34
BOX_OPENING_WIDTH_M = 0.14
BOX_EDGE_MARGIN_M = 0.04
TRANSFER_TRAVEL_Z_M = 0.300
TRANSFER_VERTICAL_STEP_M = 0.025
TRANSFER_TRANSITION_X_M = 0.45
TRANSFER_TRANSITION_Y_M = 0.05
TRANSFER_REORIENT_RPY_RAD = (math.pi, 1.0, -math.pi / 4.0)
TRANSFER_RELEASE_GRIPPER_Z_M = 0.185
TRANSFER_RETREAT_M = 0.10
TRANSFER_RETREAT_SEGMENTS = 4
TRANSFER_SEGMENT_WAYPOINTS = 7
TRANSFER_REORIENT_WAYPOINTS = 25
TRANSFER_REORIENT_SEGMENTS = 1
TRANSFER_HORIZONTAL_WAYPOINTS = 31
# Conservative visual estimate from the current handle grasp: the lowest point
# of the hanging prop is no more than 110 mm below the gripper reference point.
HELD_OBJECT_BELOW_GRIPPER_M = 0.110
MINIMUM_OBJECT_OVER_RIM_CLEARANCE_M = 0.040
MINIMUM_RELEASE_GRIPPER_OVER_RIM_M = 0.085
MAXIMUM_RELEASE_GRIPPER_OVER_RIM_M = 0.130
JOINT_LOWER_DEG = np.array([-90.0, -105.0, -90.0, -165.0, -90.0, -97.8, -172.0])
JOINT_UPPER_DEG = np.array([90.0, 105.0, 90.0, 55.0, 90.0, 102.2, 172.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robotdata",
        type=Path,
        default=Path("RobotDataCollection"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("grasp_api_tests"))
    parser.add_argument("--observation-z", type=float, default=0.40)
    parser.add_argument("--max-refinements", type=int, default=0)
    parser.add_argument("--max-refinement-camera-z", type=float, default=0.40)
    parser.add_argument(
        "--grasp-planning-attempts",
        type=int,
        default=3,
        help=(
            "repeat the stochastic GraspNet/simulation solve on the accepted "
            "near view and retain the best fully validated candidate"
        ),
    )
    parser.add_argument("--close-speed", type=int, default=200)
    parser.add_argument("--close-force", type=int, default=300)
    parser.add_argument("--target-text", default="pistol")
    parser.add_argument("--whole-object-grasp", action="store_true")
    parser.add_argument(
        "--fast-magazine",
        action="store_true",
        help="use the low-latency magazine-only perception, planning, and validation path",
    )
    parser.add_argument(
        "--relaxed-grasp-quality",
        action="store_true",
        help=(
            "relax perception/centering quality thresholds while retaining all "
            "simulation, joint, clearance, FK, and hardware-path safety gates"
        ),
    )
    parser.add_argument(
        "--resume-look-at-state",
        type=Path,
        help="resume after a completed dual-wrist look-at state artifact",
    )
    parser.add_argument(
        "--resume-refinement-plan",
        type=Path,
        help=(
            "resume from a planning-only near-view adjustment generated at the "
            "robot's current joint state"
        ),
    )
    parser.add_argument(
        "--resume-near-capture",
        type=Path,
        help="reuse a completed near RGB-D capture while the arm remains at that pose",
    )
    parser.add_argument("--execute-confirmed-inert-prop", action="store_true")
    parser.add_argument("--approve-simulated-motions", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_held_object_aperture(
    gripper_state: dict[str, Any], expected_cross_section_width_m: float
) -> dict[str, Any]:
    """Validate that the closed gripper still spans the selected object width."""
    require(
        0.0 < expected_cross_section_width_m <= MAX_GRIPPER_WIDTH_M,
        "invalid expected held-object width",
    )
    actpos = float(gripper_state.get("actpos", -1.0))
    require(0.0 <= actpos <= GRIPPER_OPEN_ACTPOS, "invalid gripper aperture state")
    estimated_aperture_m = MAX_GRIPPER_WIDTH_M * actpos / GRIPPER_OPEN_ACTPOS
    minimum_aperture_m = max(
        0.0, expected_cross_section_width_m - HOLD_APERTURE_TOLERANCE_M
    )
    report = {
        "actpos": actpos,
        "estimated_aperture_mm": estimated_aperture_m * 1000.0,
        "expected_cross_section_width_mm": expected_cross_section_width_m * 1000.0,
        "compression_tolerance_mm": HOLD_APERTURE_TOLERANCE_M * 1000.0,
        "minimum_accepted_aperture_mm": minimum_aperture_m * 1000.0,
        "passed": estimated_aperture_m >= minimum_aperture_m,
    }
    return report


def validate_magazine_hold(gripper_state: dict[str, Any]) -> dict[str, Any]:
    """Detect a non-empty magazine closure without assuming its grasped axis."""
    actpos = float(gripper_state.get("actpos", -1.0))
    current_force = float(gripper_state.get("current_force", 0.0))
    return {
        "actpos": actpos,
        "current_force": current_force,
        "minimum_nonempty_actpos": 100.0,
        "passed": actpos >= 100.0 and current_force > 0.0,
    }


def orange_component_mask(image_bgr: np.ndarray) -> np.ndarray | None:
    """Return the largest saturated orange component, if present."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([3, 110, 90], dtype=np.uint8),
        np.array([28, 255, 255], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[index, cv2.CC_STAT_AREA])
    if area < 200:
        return None
    return labels == index


def measure_wrist_target(
    gateway: Any,
    rgb_path: Path,
    depth_path: Path,
    *,
    text: str = "pistol",
) -> dict[str, Any]:
    """Measure the closest part of a segmented target in a wrist-camera frame."""
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    require(bgr is not None, f"missing wrist RGB artifact: {rgb_path}")
    depth = np.load(depth_path, allow_pickle=False)
    require(depth.shape == bgr.shape[:2], "wrist RGB/depth shapes disagree")
    rgb = bgr[:, :, ::-1].copy()
    response = gateway.open_vocabulary(
        image_bytes=rgb.tobytes(), image_shape=rgb.shape, text=text
    )
    orange_mask = orange_component_mask(bgr) if "orange" in text.casefold() else None
    masks = (
        [orange_mask]
        if orange_mask is not None
        else [
            np.squeeze(np.asarray(mask, dtype=bool))
            for mask in response["masks"]
        ]
    )
    masks = [mask for mask in masks if mask.shape == depth.shape and np.any(mask)]
    require(masks, "wrist camera could not segment the held target")
    mask = max(masks, key=np.count_nonzero)
    valid = mask & (depth > 0)
    values = depth[valid].astype(float)
    require(
        values.size >= MINIMUM_WRIST_TARGET_DEPTH_PIXELS,
        "wrist target mask has too few valid depth pixels",
    )
    rows, columns = np.nonzero(mask)
    return {
        "text": text,
        "labels": [str(value) for value in response["labels"]],
        "mask_pixels": int(mask.sum()),
        "valid_depth_pixels": int(values.size),
        "bbox_xyxy": [
            int(columns.min()),
            int(rows.min()),
            int(columns.max() + 1),
            int(rows.max() + 1),
        ],
        "centroid_uv": [float(columns.mean()), float(rows.mean())],
        "depth_mm_q10_q50_q90": [
            float(value) for value in np.quantile(values, [0.1, 0.5, 0.9])
        ],
    }


def validate_wrist_target_retention(
    reference: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Check that a target stayed near the gripper-mounted wrist camera."""
    reference_q10 = float(reference["depth_mm_q10_q50_q90"][0])
    current_q10 = float(current["depth_mm_q10_q50_q90"][0])
    near_depth_increase_mm = current_q10 - reference_q10
    report = {
        "reference": reference,
        "current": current,
        "near_depth_increase_mm": near_depth_increase_mm,
        "maximum_near_depth_increase_mm": MAXIMUM_WRIST_NEAR_DEPTH_INCREASE_MM,
        "mask_area_ratio": (
            float(current["mask_pixels"]) / float(reference["mask_pixels"])
        ),
        "passed": near_depth_increase_mm <= MAXIMUM_WRIST_NEAR_DEPTH_INCREASE_MM,
    }
    return report


def path_sha256(path: Sequence[Sequence[float]]) -> str:
    canonical = json.dumps(
        [[float(value) for value in waypoint] for waypoint in path],
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def exact_reverse_path(path: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return an independent, exact waypoint reversal with no new IK planning."""
    require(len(path) >= 2, "cannot reverse a grasp path with fewer than two waypoints")
    return [
        [float(value) for value in waypoint]
        for waypoint in reversed(path)
    ]


def rotation_error_deg(first_pose: Sequence[float], second_pose: Sequence[float]) -> float:
    first = rotation_matrix(*[float(value) for value in first_pose[3:]])
    second = rotation_matrix(*[float(value) for value in second_pose[3:]])
    cosine = float(np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=float,
    )


def plan_post_grasp_lift(
    gateway: Any,
    kinematics: Any,
    grasp_path: Sequence[Sequence[float]],
    right_joints: Sequence[float],
    preview_output_dir: Path | None = None,
    *,
    fast_mode: bool = False,
) -> dict[str, Any]:
    """Simulate and validate a world-Z lift that preserves grasp orientation."""
    require(bool(grasp_path), "cannot plan lift without a grasp path")
    start_joints = [float(value) for value in grasp_path[-1]]
    start_gripper_pose = [
        float(value) for value in kinematics.get_gripper_forward(start_joints)
    ]
    target_gripper_pose = list(start_gripper_pose)
    lift_path: list[list[float]] = []
    preview_videos = []
    current_joints = list(start_joints)
    target_end_pose: list[float] = []
    targets = []
    for segment in range(1, POST_GRASP_LIFT_SEGMENTS + 1):
        candidate_target_gripper_pose = list(start_gripper_pose)
        candidate_target_gripper_pose[2] += (
            POST_GRASP_LIFT_M * segment / POST_GRASP_LIFT_SEGMENTS
        )
        candidate_target_end_pose = [
            float(value)
            for value in kinematics.gripper_to_end(candidate_target_gripper_pose)
        ]
        target_joints = kinematics.get_gripper_inverse(
            current_joints, candidate_target_gripper_pose
        )
        require(target_joints is not None, f"post-grasp lift segment {segment} IK failed")
        target_joints = [float(value) for value in target_joints]
        target_margin_deg = float(
            np.minimum(
                np.asarray(target_joints) - JOINT_LOWER_DEG,
                JOINT_UPPER_DEG - np.asarray(target_joints),
            ).min()
        )
        if fast_mode and target_margin_deg < 0.0:
            break
        target_gripper_pose = candidate_target_gripper_pose
        target_end_pose = candidate_target_end_pose
        targets.append(
            {
                "phase": f"post_grasp_lift_{segment}",
                "pose": target_end_pose,
                "target_joints": target_joints,
                "num_waypoints": 3 if fast_mode else POST_GRASP_LIFT_WAYPOINTS_PER_SEGMENT,
            }
        )
        current_joints = target_joints

    require(bool(targets), "post-grasp lift has no reachable upward segment")

    response = gateway.simulate_sequence(
        targets=targets,
        arm="left",
        left_joints_deg=start_joints,
        right_joints_deg=right_joints,
        recording=not fast_mode,
    )
    require(bool(response["success"]), "post-grasp lift sequence simulation failed")
    require(
        len(response["segments"]) == len(targets),
        "post-grasp lift returned the wrong segment count",
    )
    current_joints = list(start_joints)
    for segment, (target, simulated) in enumerate(
        zip(targets, response["segments"]), start=1
    ):
        require(
            simulated.get("phase") == target["phase"],
            f"post-grasp lift segment {segment} phase mismatch",
        )
        segment_path = [
            [float(value) for value in waypoint] for waypoint in simulated["path"]
        ]
        require(
            2 <= len(segment_path) <= (3 if fast_mode else POST_GRASP_LIFT_WAYPOINTS_PER_SEGMENT),
            f"invalid lift segment {segment} path",
        )
        segment_start_error_deg = float(
            np.max(
                np.abs(
                    np.asarray(segment_path[0]) - np.asarray(current_joints)
                )
            )
        )
        require(
            segment_start_error_deg <= (2.0 if fast_mode else 0.1),
            f"lift segment {segment} does not start at the previous endpoint",
        )
        lift_path.extend(segment_path if not lift_path else segment_path[1:])
        current_joints = target["target_joints"]

    if not fast_mode:
        encoded_video = response.get("video")
        require(isinstance(encoded_video, str), "lift simulation omitted preview video")
        video_bytes = base64.b64decode(encoded_video, validate=True)
        preview: dict[str, Any] = {
            "phase": "full_sequence",
            "bytes": len(video_bytes),
            "sha256": hashlib.sha256(video_bytes).hexdigest(),
        }
        if preview_output_dir is not None:
            preview_output_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_output_dir / "post_grasp_lift_full_sequence.mp4"
            preview_path.write_bytes(video_bytes)
            preview["path"] = str(preview_path)
        preview_videos.append(preview)
    require(
        2
        <= len(lift_path)
        <= 1
        + len(targets)
        * ((3 if fast_mode else POST_GRASP_LIFT_WAYPOINTS_PER_SEGMENT) - 1),
        "invalid combined lift path",
    )
    start_error_deg = float(
        np.max(np.abs(np.asarray(lift_path[0]) - np.asarray(start_joints)))
    )
    require(start_error_deg <= 0.1, "lift simulation does not start at grasp endpoint")

    path_array = np.asarray(lift_path, dtype=float)
    margins = np.minimum(
        path_array.min(axis=0) - JOINT_LOWER_DEG,
        JOINT_UPPER_DEG - path_array.max(axis=0),
    )
    smallest_margin_deg = float(margins.min())
    required_margin_deg = 0.0 if fast_mode else MINIMUM_JOINT_MARGIN_DEG
    require(
        smallest_margin_deg >= required_margin_deg,
        "post-grasp lift has insufficient joint margin",
    )

    gripper_poses = [
        [float(value) for value in kinematics.get_gripper_forward(waypoint)]
        for waypoint in lift_path
    ]
    positions = np.asarray([pose[:3] for pose in gripper_poses], dtype=float)
    lateral_deviation_m = np.linalg.norm(
        positions[:, :2] - positions[0, :2], axis=1
    )
    z_progress_m = positions[:, 2] - positions[0, 2]
    require(
        float(lateral_deviation_m.max()) <= MAX_LIFT_LATERAL_DEVIATION_M,
        "post-grasp lift is not sufficiently vertical",
    )
    require(
        not np.any(np.diff(z_progress_m) < -0.002),
        "post-grasp lift does not rise monotonically",
    )
    final_position_error_m = float(
        np.linalg.norm(positions[-1] - np.asarray(target_gripper_pose[:3], dtype=float))
    )
    final_orientation_error_deg = rotation_error_deg(
        gripper_poses[-1], target_gripper_pose
    )
    require(
        final_position_error_m <= 0.005 and final_orientation_error_deg <= 3.0,
        "post-grasp lift endpoint disagrees with forward kinematics",
    )
    return {
        "distance_m": float(target_gripper_pose[2] - start_gripper_pose[2]),
        "path": lift_path,
        "path_sha256": path_sha256(lift_path),
        "waypoint_count": len(lift_path),
        "start_error_deg": start_error_deg,
        "smallest_joint_margin_deg": smallest_margin_deg,
        "maximum_lateral_deviation_mm": float(lateral_deviation_m.max() * 1000.0),
        "z_progress_mm": [float(value * 1000.0) for value in z_progress_m],
        "final_position_error_mm": final_position_error_m * 1000.0,
        "final_orientation_error_deg": final_orientation_error_deg,
        "target_gripper_pose": target_gripper_pose,
        "target_end_pose": target_end_pose,
        "preview_videos": preview_videos,
    }


def plan_box_transfer(
    gateway: Any,
    kinematics: Any,
    lifted_joints: Sequence[float],
    right_joints: Sequence[float],
    preview_output_dir: Path | None = None,
    *,
    regular_prism_mode: bool = False,
    fast_mode: bool = False,
    reorient_before_translate: bool = False,
) -> dict[str, Any]:
    """Plan a fully simulated transfer into the fixed green-box opening."""

    simulation_start_joints = [float(value) for value in lifted_joints]
    current_joints = list(simulation_start_joints)
    right_joints = [float(value) for value in right_joints]
    start_pose = [
        float(value) for value in kinematics.get_gripper_forward(current_joints)
    ]
    planned_segments: list[dict[str, Any]] = []
    preview_videos: list[dict[str, Any]] = []

    def append_segment(
        phase: str,
        target_gripper_pose: Sequence[float],
        num_waypoints: int,
    ) -> list[float]:
        nonlocal current_joints
        target = [float(value) for value in target_gripper_pose]
        target_joints = kinematics.get_gripper_inverse(current_joints, target)
        require(target_joints is not None, f"box-transfer {phase} local IK failed")
        target_joints = [float(value) for value in target_joints]
        ik_pose = [
            float(value) for value in kinematics.get_gripper_forward(target_joints)
        ]
        ik_position_error_mm = float(
            np.linalg.norm(np.asarray(ik_pose[:3]) - np.asarray(target[:3]))
            * 1000.0
        )
        ik_orientation_error_deg = rotation_error_deg(ik_pose, target)
        require(
            ik_position_error_mm <= 3.0 and ik_orientation_error_deg <= 2.0,
            f"box-transfer {phase} local IK returned the wrong endpoint",
        )
        planned_segments.append({
            "phase": phase,
            "pose": [
                float(value) for value in kinematics.gripper_to_end(target)
            ],
            "num_waypoints": int(num_waypoints),
            "target_gripper_pose": target,
            "target_joints": target_joints,
            "ik_position_error_mm": ik_position_error_mm,
            "ik_orientation_error_deg": ik_orientation_error_deg,
        })
        # Seed the next local IK from the exact validated target.  The simulator's
        # batch request will independently verify every adjacent endpoint.
        current_joints = target_joints
        return ik_pose

    # The first 100 mm lift clears the table.  Prefer the configured 300 mm
    # travel height, but retain the last reachable vertical pose if the next
    # orientation-preserving lift has no local IK solution.  The complete
    # horizontal translation, reorientation, lowering, and retreat are still
    # independently simulated.  Contact between the held prop and the box rim is
    # explicitly allowed; robot links and the gripper remain collision checked.
    pose = list(start_pose)
    require(
        float(pose[2]) < TRANSFER_TRAVEL_Z_M,
        "box-transfer start is already above travel height",
    )
    index = 0
    extra_lift_ceiling = None
    while float(pose[2]) < TRANSFER_TRAVEL_Z_M - 0.0005:
        if reorient_before_translate and index >= 3:
            extra_lift_ceiling = {
                "failed_phase": None,
                "failed_target_z_m": None,
                "highest_reachable_z_m": float(pose[2]),
                "error": "orange-box fast transfer uses the highest three validated extra lifts",
            }
            break
        index += 1
        target = list(pose)
        target[2] = min(
            TRANSFER_TRAVEL_Z_M,
            float(pose[2]) + TRANSFER_VERTICAL_STEP_M,
        )
        phase = f"extra_lift_{index}"
        try:
            pose = append_segment(
                phase,
                target,
                4 if fast_mode else TRANSFER_SEGMENT_WAYPOINTS,
            )
        except RuntimeError as error:
            if str(error) != f"box-transfer {phase} local IK failed":
                raise
            extra_lift_ceiling = {
                "failed_phase": phase,
                "failed_target_z_m": float(target[2]),
                "highest_reachable_z_m": float(pose[2]),
                "error": repr(error),
            }
            break

    high_pose = list(pose)
    minimum_safe_transfer_z_m = (
        HELD_OBJECT_BELOW_GRIPPER_M
        + BOX_RIM_Z_M
        + MINIMUM_OBJECT_OVER_RIM_CLEARANCE_M
    )
    high_object_clearance_m = (
        float(high_pose[2]) - HELD_OBJECT_BELOW_GRIPPER_M - BOX_RIM_Z_M
    )
    predicted_object_rim_overlap_m = max(0.0, -high_object_clearance_m)

    def append_reorientation(start_pose: Sequence[float]) -> list[float]:
        pose_after = list(start_pose)
        start_rpy = np.asarray(pose_after[3:], dtype=float)
        if reorient_before_translate:
            target_rpy = start_rpy.copy()
            target_rpy[2] -= math.pi / 2.0
        else:
            target_rpy = np.asarray(TRANSFER_REORIENT_RPY_RAD, dtype=float)
        wrapped_rpy_delta = (
            target_rpy - start_rpy + math.pi
        ) % (2.0 * math.pi) - math.pi
        segment_count = (
            4
            if reorient_before_translate
            else 2
            if regular_prism_mode
            else TRANSFER_REORIENT_SEGMENTS
        )
        for index in range(1, segment_count + 1):
            reorient_target = list(start_pose)
            reorient_target[3:] = (
                start_rpy + wrapped_rpy_delta * index / segment_count
            ).tolist()
            pose_after = append_segment(
                f"stationary_reorient_{index}",
                reorient_target,
                4 if fast_mode else TRANSFER_SEGMENT_WAYPOINTS,
            )
        return pose_after

    if reorient_before_translate:
        # Grasp the medicine box across its short side first, then rotate the
        # already-held object as a rigid body into the transfer-friendly wrist
        # orientation.  This never substitutes a long-side grasp.
        pose = append_reorientation(high_pose)

    transition_target = list(pose)
    transition_target[0] = TRANSFER_TRANSITION_X_M
    transition_target[1] = TRANSFER_TRANSITION_Y_M
    pose = append_segment(
        "horizontal_to_transition",
        transition_target,
        12 if fast_mode else TRANSFER_HORIZONTAL_WAYPOINTS,
    )

    if not reorient_before_translate:
        pose = append_reorientation(pose)

    box_drop_x_m = 0.32 if reorient_before_translate else BOX_DROP_X_M
    box_drop_y_m = -0.145 if reorient_before_translate else BOX_DROP_Y_M
    transfer_target = list(pose)
    transfer_target[0] = box_drop_x_m
    transfer_target[1] = box_drop_y_m
    pose = append_segment(
        "above_box", transfer_target, 12 if fast_mode else TRANSFER_HORIZONTAL_WAYPOINTS
    )

    minimum_release_over_rim_m = (
        0.050 if reorient_before_translate else MINIMUM_RELEASE_GRIPPER_OVER_RIM_M
    )
    effective_release_gripper_z_m = max(
        BOX_RIM_Z_M + minimum_release_over_rim_m,
        min(
            TRANSFER_RELEASE_GRIPPER_Z_M,
            float(transfer_target[2]) - (0.005 if fast_mode else 0.0),
        ),
    )
    lower_distance_m = float(transfer_target[2]) - effective_release_gripper_z_m
    require(lower_distance_m > 0.0, "configured green-box release is above travel height")
    lower_segment_count = int(math.ceil(lower_distance_m / TRANSFER_VERTICAL_STEP_M))
    for index in range(1, lower_segment_count + 1):
        target = list(transfer_target)
        target[2] -= lower_distance_m * index / lower_segment_count
        pose = append_segment(
            f"lower_into_box_{index}", target, 4 if fast_mode else TRANSFER_SEGMENT_WAYPOINTS
        )

    release_pose = list(pose)
    release_over_rim_m = float(release_pose[2]) - BOX_RIM_Z_M
    require(
        minimum_release_over_rim_m
        <= release_over_rim_m
        <= MAXIMUM_RELEASE_GRIPPER_OVER_RIM_M,
        "green-box release height is outside the configured drop window",
    )

    pre_release_count = len(planned_segments)
    retreat_start_pose = list(release_pose)
    retreat_achieved_m = 0.0
    retreat_ceiling = None
    for index in range(1, TRANSFER_RETREAT_SEGMENTS + 1):
        target = list(retreat_start_pose)
        target[2] += TRANSFER_RETREAT_M * index / TRANSFER_RETREAT_SEGMENTS
        phase = f"post_release_retreat_{index}"
        try:
            append_segment(
                phase, target, 4 if fast_mode else TRANSFER_SEGMENT_WAYPOINTS
            )
            retreat_achieved_m = float(target[2] - retreat_start_pose[2])
        except RuntimeError as error:
            if not reorient_before_translate or str(error) != (
                f"box-transfer {phase} local IK failed"
            ):
                raise
            retreat_ceiling = {
                "failed_phase": phase,
                "highest_reachable_z_m": float(
                    retreat_start_pose[2] + retreat_achieved_m
                ),
                "error": repr(error),
            }
            break
    if not reorient_before_translate:
        require(
            retreat_achieved_m >= TRANSFER_RETREAT_M / TRANSFER_RETREAT_SEGMENTS,
            "box-transfer has no reachable post-release retreat",
        )
    response = gateway.simulate_sequence(
        targets=planned_segments,
        arm="left",
        left_joints_deg=simulation_start_joints,
        right_joints_deg=right_joints,
        recording=preview_output_dir is not None,
    )
    require(bool(response["success"]), "box-transfer sequence simulation failed")
    simulated_segments = response["segments"]
    require(
        len(simulated_segments) == len(planned_segments),
        "box-transfer sequence returned the wrong number of segments",
    )

    segments: list[dict[str, Any]] = []
    expected_start = simulation_start_joints
    for planned, simulated in zip(planned_segments, simulated_segments):
        phase = str(planned["phase"])
        require(
            simulated.get("phase") == phase,
            f"box-transfer sequence phase mismatch at {phase}",
        )
        path = [
            [float(value) for value in waypoint]
            for waypoint in simulated["path"]
        ]
        require(
            2 <= len(path) <= (200 if fast_mode else int(planned["num_waypoints"])),
            f"invalid {phase} path",
        )
        if fast_mode and len(path) > int(planned["num_waypoints"]):
            indices = np.linspace(
                0,
                len(path) - 1,
                int(planned["num_waypoints"]),
            ).round().astype(int)
            path = [path[int(index)] for index in indices]
        raw_start_error_deg = float(
            np.max(np.abs(np.asarray(path[0]) - np.asarray(expected_start)))
        )
        inserted_start_bridge = False
        if (
            reorient_before_translate
            and fast_mode
            and 2.0 < raw_start_error_deg <= 3.2
        ):
            path.insert(0, [float(value) for value in expected_start])
            inserted_start_bridge = True
        start_error_deg = float(
            np.max(np.abs(np.asarray(path[0]) - np.asarray(expected_start)))
        )
        require(
            start_error_deg <= (2.0 if fast_mode else 0.1),
            (
                f"box-transfer {phase} does not start at the previous endpoint: "
                f"{start_error_deg:.3f} deg"
            ),
        )
        reached_pose = [
            float(value) for value in kinematics.get_gripper_forward(path[-1])
        ]
        joint_target_error_deg = float(
            np.max(
                np.abs(
                    np.asarray(path[-1]) - np.asarray(planned["target_joints"])
                )
            )
        )
        position_error_mm = float(
            np.linalg.norm(
                np.asarray(reached_pose[:3])
                - np.asarray(planned["target_gripper_pose"][:3])
            )
            * 1000.0
        )
        orientation_error_deg = rotation_error_deg(
            reached_pose, planned["target_gripper_pose"]
        )
        require(
            joint_target_error_deg <= 0.1
            and position_error_mm <= 6.0
            and orientation_error_deg <= 3.0,
            f"box-transfer {phase} endpoint disagrees with forward kinematics",
        )
        segments.append(
            {
                "phase": phase,
                "path": path,
                "path_sha256": path_sha256(path),
                "waypoint_count": len(path),
                "start_error_deg": start_error_deg,
                "raw_start_error_deg": raw_start_error_deg,
                "inserted_start_bridge": inserted_start_bridge,
                "target_gripper_pose": planned["target_gripper_pose"],
                "target_joints": planned["target_joints"],
                "reached_gripper_pose": reached_pose,
                "joint_target_error_deg": joint_target_error_deg,
                "ik_position_error_mm": planned["ik_position_error_mm"],
                "ik_orientation_error_deg": planned["ik_orientation_error_deg"],
                "position_error_mm": position_error_mm,
                "orientation_error_deg": orientation_error_deg,
            }
        )
        expected_start = planned["target_joints"]

    if preview_output_dir is not None:
        encoded_video = response.get("video")
        require(
            isinstance(encoded_video, str),
            "box-transfer sequence simulation omitted preview video",
        )
        video_bytes = base64.b64decode(encoded_video, validate=True)
        preview_output_dir.mkdir(parents=True, exist_ok=True)
        video_path = preview_output_dir / "box_transfer_full_sequence.mp4"
        video_path.write_bytes(video_bytes)
        preview_videos.append(
            {
                "phase": "full_sequence",
                "bytes": len(video_bytes),
                "sha256": hashlib.sha256(video_bytes).hexdigest(),
                "path": str(video_path),
            }
        )

    pre_release_segments = segments[:pre_release_count]
    retreat_segments = segments[pre_release_count:]
    pre_release_path: list[list[float]] = []
    for segment in pre_release_segments:
        path = segment["path"]
        pre_release_path.extend(path if not pre_release_path else path[1:])
    retreat_path: list[list[float]] = []
    for segment in retreat_segments:
        path = segment["path"]
        retreat_path.extend(path if not retreat_path else path[1:])

    all_paths = [*pre_release_path, *retreat_path[1:]]
    path_array = np.asarray(all_paths, dtype=float)
    margins = np.minimum(
        path_array.min(axis=0) - JOINT_LOWER_DEG,
        JOINT_UPPER_DEG - path_array.max(axis=0),
    )
    smallest_margin_deg = float(margins.min())
    required_margin_deg = 0.0 if fast_mode else MINIMUM_JOINT_MARGIN_DEG
    require(
        smallest_margin_deg >= required_margin_deg,
        f"box-transfer path has insufficient joint margin: {smallest_margin_deg:.3f} deg",
    )

    return {
        "box": {
            "drop_xy_m": [box_drop_x_m, box_drop_y_m],
            "rim_z_m": BOX_RIM_Z_M,
            "opening_length_m": BOX_OPENING_LENGTH_M,
            "opening_width_m": BOX_OPENING_WIDTH_M,
            "edge_margin_m": BOX_EDGE_MARGIN_M,
        },
        "start_gripper_pose": start_pose,
        "high_gripper_pose": high_pose,
        "requested_travel_z_m": TRANSFER_TRAVEL_Z_M,
        "achieved_travel_z_m": float(high_pose[2]),
        "minimum_safe_transfer_z_m": minimum_safe_transfer_z_m,
        "extra_lift_ceiling": extra_lift_ceiling,
        "release_gripper_pose": release_pose,
        "retreat_achieved_m": retreat_achieved_m,
        "retreat_ceiling": retreat_ceiling,
        "high_object_over_rim_clearance_m": high_object_clearance_m,
        "predicted_object_rim_overlap_m": predicted_object_rim_overlap_m,
        "held_object_box_rim_contact_allowed": True,
        "orientation_policy": (
            "short_side_grasp_then_stationary_90deg_yaw"
            if reorient_before_translate
            else "two_stage_stationary_reorient_for_magazine"
            if regular_prism_mode
            else "stationary_reorient_for_pistol"
        ),
        "release_gripper_over_rim_m": release_over_rim_m,
        "pre_release_segments": pre_release_segments,
        "pre_release_path": pre_release_path,
        "pre_release_path_sha256": path_sha256(pre_release_path),
        "retreat_segments": retreat_segments,
        "retreat_path": retreat_path,
        "retreat_path_sha256": path_sha256(retreat_path),
        "smallest_joint_margin_deg": smallest_margin_deg,
        "preview_videos": preview_videos,
    }


def run_child(command: Sequence[str], *, cwd: Path, log: Path) -> dict[str, Any]:
    print("RUN", " ".join(command), flush=True)
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="", flush=True)
    require(completed.returncode == 0, f"child command failed ({completed.returncode})")
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("child command produced no JSON result")


def run_read_only_child_with_retry(
    command: Sequence[str], *, cwd: Path, log: Path
) -> dict[str, Any]:
    """Retry a read-only capture up to four times; never retry robot motion."""
    maximum_attempts = 4
    errors = []
    for attempt in range(maximum_attempts):
        attempt_log = (
            log
            if attempt == 0
            else log.with_name(f"{log.stem}_retry{attempt}{log.suffix}")
        )
        try:
            return run_child(command, cwd=cwd, log=attempt_log)
        except RuntimeError as error:
            errors.append(repr(error))
            print(
                f"READ_ONLY_RETRY attempt={attempt + 1}/{maximum_attempts} "
                f"error={error!r}",
                flush=True,
            )
            if attempt + 1 == maximum_attempts:
                raise RuntimeError(
                    f"read-only child failed after {maximum_attempts} attempts: "
                    + "; ".join(errors)
                ) from error
            time.sleep(float(attempt + 1))
    raise AssertionError("unreachable read-only retry state")


def resolve_output(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def capture_world_points(directory: Path) -> np.ndarray:
    mask = np.load(directory / "barrel_sam_mask.npy", allow_pickle=False).astype(bool)
    depth = np.load(directory / "left_depth.npy", allow_pickle=False)
    plan = json.loads((directory / "plan.json").read_text("utf-8"))
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    valid = eroded & (depth > 0)
    values = depth[valid].astype(float)
    require(values.size >= 500, "too few valid barrel depth pixels")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    valid &= np.abs(depth.astype(float) - median) <= max(3.0 * 1.4826 * mad, 8.0)
    rows, columns = np.nonzero(valid)
    z = depth[valid].astype(float) / 1000.0
    ppx, ppy = 308.3497314453125, 239.98605346679688
    fx, fy = 607.5458374023438, 607.83642578125
    camera = np.column_stack(
        ((columns - ppx) * z / fx, (rows - ppy) * z / fy, z)
    )
    camera_pose = [float(value) for value in plan["current_camera_pose"]]
    return camera @ rotation_matrix(*camera_pose[3:]).T + np.asarray(camera_pose[:3])


def strict_candidates(
    directory: Path,
    evaluation: dict[str, Any],
    kinematics: Any,
    *,
    relaxed_quality: bool = False,
    regular_prism_mode: bool = False,
    fast_mode: bool = False,
    minimum_table_clearance_m: float = MINIMUM_TABLE_CLEARANCE_M,
) -> list[dict[str, Any]]:
    points_world = capture_world_points(directory)
    plan = json.loads((directory / "plan.json").read_text("utf-8"))
    seed = [float(value) for value in plan["left_joints"]]
    simulations = {int(item["rank"]): item for item in evaluation["simulations"]}
    results = []
    for candidate in evaluation["validated_candidates"]:
        rank = int(candidate["rank"])
        result: dict[str, Any] = {
            "rank": rank,
            "candidate_source": candidate.get("candidate_source", "graspnet"),
            "accepted": False,
            "reasons": [],
            "quality_warnings": [],
            "quality_policy": "relaxed" if relaxed_quality else "strict",
        }
        simulation = simulations.get(rank)
        if not candidate.get("plausible"):
            destination = "quality_warnings" if relaxed_quality else "reasons"
            result[destination].append("broad perception filter failed")
        if not simulation or not simulation.get("stage1_success") or not simulation.get(
            "stage2_success"
        ):
            result["reasons"].append("two-stage simulation failed")

        pose = [float(value) for value in candidate["pose"]]
        left_arm_base_distance_m = float(
            candidate.get(
                "left_arm_base_distance_m",
                np.linalg.norm(np.asarray(pose[:3]) - np.array([0.0, 0.25, 0.0])),
            )
        )
        result["left_arm_base_distance_m"] = left_arm_base_distance_m
        grasp_rotation = rotation_matrix(*pose[3:])
        local = (points_world - np.asarray(pose[:3])) @ grasp_rotation
        sliced = local[(np.abs(local[:, 2]) <= 0.008) & (np.abs(local[:, 0]) <= 0.012), 1]
        minimum_cross_section_points = (
            MAGAZINE_MINIMUM_CROSS_SECTION_POINTS if regular_prism_mode else 200
        )
        result["minimum_cross_section_points"] = minimum_cross_section_points
        if sliced.size < minimum_cross_section_points:
            result["reasons"].append("too few local 3D cross-section points")
            low = high = float("nan")
        else:
            low, high = (float(value) for value in np.quantile(sliced, [0.03, 0.97]))
            width = high - low
            half_opening = MAX_GRIPPER_WIDTH_M / 2.0
            side_clearance = min(low + half_opening, half_opening - high)
            result.update(
                {
                    "cross_section_interval_mm": [low * 1000.0, high * 1000.0],
                    "cross_section_width_mm": width * 1000.0,
                    "worst_side_clearance_mm": side_clearance * 1000.0,
                }
            )
            if regular_prism_mode:
                orange_box_mode = (
                    minimum_table_clearance_m < MINIMUM_TABLE_CLEARANCE_M
                )
                width_limits = (
                    (MAGAZINE_CROSS_SECTION_WIDTH_M[0], MAX_GRIPPER_WIDTH_M)
                    if orange_box_mode
                    else MAGAZINE_CROSS_SECTION_WIDTH_M
                )
                minimum_side_clearance = (
                    -0.010 if orange_box_mode else MAGAZINE_MINIMUM_SIDE_CLEARANCE_M
                )
            else:
                width_limits = (
                    RELAXED_CROSS_SECTION_WIDTH_M
                    if relaxed_quality
                    else STRICT_CROSS_SECTION_WIDTH_M
                )
                minimum_side_clearance = (
                    RELAXED_MINIMUM_SIDE_CLEARANCE_M
                    if relaxed_quality
                    else STRICT_MINIMUM_SIDE_CLEARANCE_M
                )
            result["quality_width_limits_mm"] = [
                value * 1000.0 for value in width_limits
            ]
            result["quality_minimum_side_clearance_mm"] = (
                minimum_side_clearance * 1000.0
            )
            if not width_limits[0] <= width <= width_limits[1]:
                result["reasons"].append("3D cross-section width is outside limits")
            if side_clearance < minimum_side_clearance:
                result["reasons"].append(
                    "insufficient physical opening clearance on one side"
                )

        selected_advance_m = (
            float(simulation["selected_advance_m"])
            if simulation and simulation.get("stage2_success")
            else 0.0
        )
        result["selected_advance_m"] = selected_advance_m
        requested_grasp_depth_m = float(candidate.get("gdepth_m", 0.0))
        result["requested_grasp_depth_mm"] = requested_grasp_depth_m * 1000.0
        result["insertion_beyond_graspnet_mm"] = (
            selected_advance_m - requested_grasp_depth_m
        ) * 1000.0
        pregrasp_retraction_m = (
            float(simulation.get("pregrasp_retraction_m", float("nan")))
            if simulation
            else float("nan")
        )
        result["pregrasp_retraction_mm"] = pregrasp_retraction_m * 1000.0
        result["required_pregrasp_retraction_mm"] = (
            REQUIRED_PREGRASP_RETRACTION_M * 1000.0
        )
        if not math.isclose(
            pregrasp_retraction_m,
            REQUIRED_PREGRASP_RETRACTION_M,
            abs_tol=1e-9,
        ):
            result["reasons"].append("simulation did not use the required pregrasp retraction")
        final_gripper_pose = kinematics.cal_pre_pose(pose, -selected_advance_m)
        endpoint_clearance_m = virtual_tip_clearance_m(final_gripper_pose)
        result["endpoint_virtual_tip_clearance_mm"] = endpoint_clearance_m * 1000.0
        result["virtual_collision_length_mm"] = VIRTUAL_COLLISION_LENGTH_M * 1000.0
        result["minimum_table_clearance_mm"] = minimum_table_clearance_m * 1000.0
        if endpoint_clearance_m < minimum_table_clearance_m:
            result["reasons"].append("final virtual tip does not meet table clearance")

        simulation_path = simulation.get("waypoints", []) if simulation else []
        if not simulation_path:
            result["reasons"].append("simulation omitted executable waypoints")
        else:
            simulation_path = [
                [float(value) for value in waypoint] for waypoint in simulation_path
            ]
            path_array = np.asarray(simulation_path, dtype=float)
            result["simulation_waypoints"] = simulation_path
            result["simulation_path_sha256"] = path_sha256(simulation_path)
            margins = np.minimum(
                path_array.min(axis=0) - JOINT_LOWER_DEG,
                JOINT_UPPER_DEG - path_array.max(axis=0),
            )
            result["smallest_joint_margin_deg"] = float(margins.min())
            required_joint_margin_deg = 0.0 if fast_mode else MINIMUM_JOINT_MARGIN_DEG
            result["required_joint_margin_deg"] = required_joint_margin_deg
            if float(margins.min()) < MINIMUM_JOINT_MARGIN_DEG:
                if not fast_mode or float(margins.min()) < required_joint_margin_deg:
                    result["reasons"].append(
                        f"simulated path has less than {required_joint_margin_deg:g} degree margin"
                    )

            waypoint_clearances_m = []
            for waypoint in simulation_path:
                tcp_pose = kinematics.get_gripper_forward(waypoint)
                waypoint_clearances_m.append(virtual_tip_clearance_m(tcp_pose))
            minimum_path_clearance_m = min(waypoint_clearances_m)
            result["minimum_path_virtual_tip_clearance_mm"] = (
                minimum_path_clearance_m * 1000.0
            )
            result["waypoint_virtual_tip_clearances_mm"] = [
                value * 1000.0 for value in waypoint_clearances_m
            ]
            if minimum_path_clearance_m < minimum_table_clearance_m:
                result["reasons"].append(
                    "simulated waypoint path does not meet virtual-tip table clearance"
                )

            stage1_waypoint_count = int(simulation.get("stage1_waypoint_count", 0))
            result["stage1_waypoint_count"] = stage1_waypoint_count
            if not 0 < stage1_waypoint_count < len(simulation_path):
                result["reasons"].append("simulation omitted the pregrasp/approach boundary")
            else:
                wrist7_to_pregrasp_delta_deg = abs(
                    simulation_path[stage1_waypoint_count - 1][6]
                    - simulation_path[0][6]
                )
                result["wrist7_to_pregrasp_delta_deg"] = (
                    wrist7_to_pregrasp_delta_deg
                )
                maximum_wrist7_delta_deg = (
                    170.0
                    if fast_mode
                    else MAGAZINE_MAX_WRIST7_TO_PREGRASP_DELTA_DEG
                    if regular_prism_mode
                    else MAX_WRIST7_TO_PREGRASP_DELTA_DEG
                )
                result["maximum_wrist7_to_pregrasp_delta_deg"] = (
                    maximum_wrist7_delta_deg
                )
                if wrist7_to_pregrasp_delta_deg > maximum_wrist7_delta_deg:
                    result["reasons"].append(
                        "wrist-7 rotation to pregrasp exceeds limit"
                    )
                expected_pregrasp_pose = kinematics.cal_pre_pose(
                    pose, REQUIRED_PREGRASP_RETRACTION_M
                )
                approach_axis = rotation_matrix(*pose[3:])[:, 0]
                approach_positions = [
                    np.asarray(expected_pregrasp_pose[:3], dtype=float)
                ]
                approach_positions.extend(
                    np.asarray(kinematics.get_gripper_forward(waypoint)[:3], dtype=float)
                    for waypoint in simulation_path[stage1_waypoint_count:]
                )
                approach_deltas = np.asarray(approach_positions) - approach_positions[0]
                approach_progress_m = approach_deltas @ approach_axis
                lateral_vectors = approach_deltas - np.outer(
                    approach_progress_m, approach_axis
                )
                lateral_deviations_m = np.linalg.norm(lateral_vectors, axis=1)
                result["approach_progress_mm"] = [
                    float(value * 1000.0) for value in approach_progress_m
                ]
                result["approach_lateral_deviations_mm"] = [
                    float(value * 1000.0) for value in lateral_deviations_m
                ]
                result["maximum_approach_lateral_deviation_mm"] = float(
                    lateral_deviations_m.max() * 1000.0
                )
                result["allowed_approach_lateral_deviation_mm"] = (
                    MAX_APPROACH_LATERAL_DEVIATION_M * 1000.0
                )
                if float(lateral_deviations_m.max()) > MAX_APPROACH_LATERAL_DEVIATION_M:
                    result["reasons"].append(
                        "pregrasp-to-final path is not sufficiently straight"
                    )
                if np.any(np.diff(approach_progress_m) < -0.002):
                    result["reasons"].append(
                        "pregrasp-to-final path does not advance monotonically"
                    )

            actual_final_pose = [
                float(value) for value in kinematics.get_gripper_forward(simulation_path[-1])
            ]
            position_error_m = float(
                np.linalg.norm(
                    np.asarray(actual_final_pose[:3])
                    - np.asarray(final_gripper_pose[:3], dtype=float)
                )
            )
            orientation_error_deg = rotation_error_deg(
                actual_final_pose, final_gripper_pose
            )
            result["realman_fk_final_position_error_mm"] = position_error_m * 1000.0
            result["realman_fk_final_orientation_error_deg"] = orientation_error_deg
            if position_error_m > 0.005 or orientation_error_deg > 3.0:
                result["reasons"].append(
                    "simulated path final pose disagrees with RealMan forward kinematics"
                )

        legacy_path, _ = kinematics.plan_grasp(
            seed, [(pose, selected_advance_m)]
        )
        result["legacy_ik_success"] = bool(legacy_path)
        if legacy_path:
            result["legacy_waypoints_diagnostic_only"] = legacy_path

        result["accepted"] = not result["reasons"]
        if result["accepted"]:
            preferred_physical_grasp = (
                float(result["worst_side_clearance_mm"])
                >= PREFERRED_MINIMUM_SIDE_CLEARANCE_M * 1000.0
            )
            result["preferred_physical_grasp"] = preferred_physical_grasp
            result["preferred_minimum_side_clearance_mm"] = (
                PREFERRED_MINIMUM_SIDE_CLEARANCE_M * 1000.0
            )
            # Retain marginal relaxed candidates as a no-stop fallback.  Prefer
            # a physically centered grasp tier first, then the nearer pose in
            # that tier to avoid reaching across the trigger region.
            if regular_prism_mode:
                # The magazine is a regular prism, so distance to the arm base
                # is not a semantic preference.  Prefer the analytic centre
                # grasp, balanced finger clearance, and ample joint margin.
                result["selection_score"] = (
                    (2.0 if result["candidate_source"] == "analytic_topdown_prism" else 0.0)
                    + 0.01 * float(result.get("worst_side_clearance_mm", -100.0))
                    + 0.005 * float(result.get("smallest_joint_margin_deg", 0.0))
                )
                result["selection_policy"] = (
                    "magazine_centered_topdown_clearance_and_joint_margin"
                )
            else:
                result["selection_score"] = (
                    (1.0 if preferred_physical_grasp else 0.0)
                    - left_arm_base_distance_m
                )
                result["selection_policy"] = "pistol_near_base_tier_then_distance"
        results.append(result)
    return results


def save_observation(observation: dict[str, Any], output: Path, prefix: str) -> None:
    for name in ("left", "head", "right"):
        if name not in observation:
            continue
        rgb = np.asarray(observation[name]["rgb"])
        depth = np.asarray(observation[name]["depth"])
        cv2.imwrite(str(output / f"{prefix}_{name}_rgb.jpg"), rgb[:, :, ::-1])
        np.save(output / f"{prefix}_{name}_depth.npy", depth, allow_pickle=False)


def save_recorded_observation(
    recorder: HardwareVideoRecorder,
    output: Path,
    prefix: str,
) -> dict[str, str]:
    requested = recorder.mark(prefix)
    observation, camera_errors = recorder.snapshot_after(requested)
    save_observation(observation, output, prefix)
    return camera_errors


def near_view_quality(
    directory: Path, view: dict[str, Any]
) -> tuple[tuple[float, float, int], dict[str, Any]]:
    """Rank usable captures without turning image quality into a safety bypass."""
    image = cv2.imread(str(directory / "left_rgb.jpg"), cv2.IMREAD_COLOR)
    require(image is not None, "near-view RGB artifact is missing")
    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = [float(value) for value in view["bbox_xyxy"]]
    require(
        0.0 <= x_min < x_max <= float(width)
        and 0.0 <= y_min < y_max <= float(height),
        "near-view bbox lies outside the captured image",
    )
    margins = (
        x_min / width,
        y_min / height,
        (width - x_max) / width,
        (height - y_max) / height,
    )
    clipped_edges = sum(value <= (1.0 / max(width, height)) for value in margins)
    minimum_margin = min(margins)
    valid_depth_points = int(view.get("sam_mask_world_point_count", 0))
    require(valid_depth_points >= 500, "near view has too few valid target depth points")
    details = {
        "directory": str(directory),
        "clipped_edges": clipped_edges,
        "minimum_normalized_margin": minimum_margin,
        "valid_target_depth_points": valid_depth_points,
        "bbox_xyxy": [x_min, y_min, x_max, y_max],
    }
    return (-float(clipped_edges), minimum_margin, valid_depth_points), details


def main() -> int:
    arguments = parse_args()
    is_orange_box = "orange" in arguments.target_text.casefold()
    require(
        not arguments.fast_magazine or arguments.whole_object_grasp,
        "--fast-magazine requires --whole-object-grasp",
    )
    require(arguments.execute_confirmed_inert_prop, "inert-prop confirmation flag required")
    require(arguments.approve_simulated_motions, "simulated-motion approval flag required")
    require(0.27 <= arguments.observation_z <= 0.40, "invalid observation height")
    require(0 <= arguments.max_refinements <= 8, "max refinements must be 0..8")
    require(
        1 <= arguments.grasp_planning_attempts <= 5,
        "grasp planning attempts must be 1..5",
    )
    require(
        arguments.observation_z <= arguments.max_refinement_camera_z <= 0.40,
        "refinement camera-z limit must be between observation z and 0.40 m",
    )
    require(1 <= arguments.close_speed <= 1000, "invalid close speed")
    require(1 <= arguments.close_force <= 1000, "invalid close force")
    require(
        not (
            arguments.resume_look_at_state is not None
            and arguments.resume_refinement_plan is not None
        ),
        "choose only one resume artifact",
    )

    repo = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_slug = "".join(
        character if character.isalnum() else "_"
        for character in arguments.target_text.lower()
    ).strip("_")
    require(bool(target_slug), "target text must contain an alphanumeric character")
    session = arguments.output_root / f"auto_{target_slug}_{stamp}"
    if not session.is_absolute():
        session = repo / session
    captures = session / "captures"
    captures.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "ok": False,
        "session": str(session),
        "real_robot_motion_sent": False,
        "gripper_close_sent": False,
        "lift_sent": False,
        "lift_completed": False,
        "box_transfer_sent": False,
        "box_release_sent": False,
        "box_retreat_completed": False,
        "grasp_quality_policy": (
            "relaxed" if arguments.relaxed_grasp_quality else "strict"
        ),
        "minimum_joint_margin_deg": MINIMUM_JOINT_MARGIN_DEG,
        "target_text": arguments.target_text,
        "whole_object_grasp": bool(arguments.whole_object_grasp),
        "events": [],
    }
    environment = None
    selected_arm = None
    hardware_video = None
    try:
        python = sys.executable
        common_capture = [
            python,
            str(repo / "scripts/plan_pistol_observation.py"),
            "--robotdata",
            str(arguments.robotdata),
            "--output-root",
            str(captures),
            "--arm",
            "left",
            "--text",
            arguments.target_text,
            "--api-host",
            "127.0.0.1",
            "--max-camera-z",
            str(arguments.max_refinement_camera_z),
        ]

        refinement_start_iteration = 0
        near_directory = None
        if arguments.resume_near_capture is not None:
            near_directory = arguments.resume_near_capture.resolve()
            require(near_directory.is_dir(), "resume near capture does not exist")
            require((near_directory / "plan.json").is_file(), "resume near capture has no plan")
            require((near_directory / "left_rgb.jpg").is_file(), "resume near capture has no RGB")
            require((near_directory / "left_depth.npy").is_file(), "resume near capture has no depth")
            require((near_directory / "target_mask.npy").is_file(), "resume near capture has no mask")
            state["events"].append(
                {"stage": "resume_near_capture", "directory": str(near_directory)}
            )
            refinement_start_iteration = arguments.max_refinements + 1
        elif arguments.resume_refinement_plan is not None:
            refinement_plan = arguments.resume_refinement_plan.resolve()
            require(refinement_plan.is_file(), "resume refinement plan does not exist")
            resumed = json.loads(refinement_plan.read_text("utf-8"))
            require(resumed.get("ok") is True, "resume refinement plan is not successful")
            require(
                resumed.get("planning_only") is True,
                "resume refinement artifact is not planning-only",
            )
            require(
                resumed.get("real_robot_command_sent") is False,
                "resume refinement artifact already sent robot motion",
            )
            require(
                resumed.get("gripper_command_sent") is False,
                "resume refinement artifact used gripper",
            )
            require(
                resumed.get("status") == "awaiting_human_review",
                "resume refinement artifact has no simulated motion",
            )
            require(resumed.get("arm") == "left", "resume refinement plan is not for left arm")
            run_child(
                [
                    python,
                    str(repo / "scripts/execute_observation_plan.py"),
                    str(refinement_plan),
                    "--robotdata",
                    str(arguments.robotdata),
                    "--execute-approved-observation",
                ],
                cwd=repo,
                log=session / "00_resume_refinement_execute.log",
            )
            state["real_robot_motion_sent"] = True
            state["events"].append(
                {"stage": "resume_refinement_execute", "plan": str(refinement_plan)}
            )
            refinement_start_iteration = 1
        elif arguments.resume_look_at_state is not None:
            look_at_state = arguments.resume_look_at_state.resolve()
            require(look_at_state.is_file(), "resume look-at state does not exist")
            resumed = json.loads(look_at_state.read_text("utf-8"))
            require(resumed.get("ok") is True, "resume look-at state is not successful")
            require(resumed.get("stage") == "look-at", "resume artifact is not look-at")
            require(resumed.get("gripper_command_sent") is False, "resume artifact used gripper")
            state["events"].append(
                {"stage": "resume_after_dual_wrist_look_at", "state": str(look_at_state)}
            )
        else:
            initial_view = run_child(
                [
                    python,
                    str(repo / "scripts/move_dual_wrist_observation.py"),
                    "initial",
                    "--robotdata",
                    str(arguments.robotdata),
                    "--output-root",
                    str(captures),
                    "--execute-approved-observation",
                    "--release-left-before-initial",
                ],
                cwd=repo,
                log=session / "01_initial_dual_wrist_view.log",
            )
            require(initial_view.get("ok") is True, "initial dual-wrist view failed")
            state["real_robot_motion_sent"] = True
            state["events"].append({"stage": "initial_dual_wrist_view"})

            far = run_read_only_child_with_retry(
                [
                    python,
                    str(repo / "scripts/capture_dual_wrist_target.py"),
                    "--robotdata",
                    str(arguments.robotdata),
                    "--output-root",
                    str(captures),
                    "--arm",
                    "left",
                    "--text",
                    arguments.target_text,
                ],
                cwd=repo,
                log=session / "02_far_dual_wrist_capture.log",
            )
            require(far.get("ok") is True, "far capture/detection failed")
            far_directory = resolve_output(repo, str(far["output"]))
            state["events"].append(
                {
                    "stage": "far_dual_wrist_capture",
                    "output": str(far_directory),
                    "fusion": far.get("fusion"),
                    "head_used_for_geometry": False,
                }
            )

            look_at = run_child(
                [
                    python,
                    str(repo / "scripts/move_dual_wrist_observation.py"),
                    "look-at",
                    "--center-plan",
                    str(far_directory / "plan.json"),
                    "--robotdata",
                    str(arguments.robotdata),
                    "--output-root",
                    str(captures),
                    "--execute-approved-observation",
                ],
                cwd=repo,
                log=session / "03_dual_wrist_look_at.log",
            )
            require(look_at.get("ok") is True, "dual-wrist look-at failed")
            look_at_state = resolve_output(repo, str(look_at["state"]))
            state["events"].append({"stage": "dual_wrist_look_at"})

        if (
            arguments.resume_refinement_plan is None
            and arguments.resume_near_capture is None
        ):
            centered = run_child(
                [
                    python,
                    str(repo / "scripts/plan_mask_centered_observation.py"),
                    str(look_at_state),
                    "--observation-z",
                    str(arguments.observation_z),
                    "--robotdata",
                    str(arguments.robotdata),
                ],
                cwd=repo,
                log=session / "04_overlook_plan.log",
            )
            centered_plan = resolve_output(repo, str(centered["plan"]))
            run_child(
                [
                    python,
                    str(repo / "scripts/execute_observation_plan.py"),
                    str(centered_plan),
                    "--robotdata",
                    str(arguments.robotdata),
                    "--execute-approved-observation",
                ],
                cwd=repo,
                log=session / "05_overlook_execute.log",
            )
            state["real_robot_motion_sent"] = True
            state["events"].append({"stage": "mask_centered_overlook"})

        usable_near_views: list[
            tuple[tuple[float, float, int], Path, dict[str, Any]]
        ] = []
        for iteration in range(
            refinement_start_iteration, arguments.max_refinements + 1
        ):
            view = run_read_only_child_with_retry(
                common_capture,
                cwd=repo,
                log=session / f"06_refinement_{iteration}_capture.log",
            )
            require(view.get("ok") is True, f"refinement capture {iteration} failed")
            view_directory = resolve_output(repo, str(view["output"]))
            quality, quality_details = near_view_quality(view_directory, view)
            usable_near_views.append((quality, view_directory, quality_details))
            state["events"].append(
                {
                    "stage": "refinement_capture",
                    "iteration": iteration,
                    "status": view.get("status"),
                    "bbox": view.get("bbox_xyxy"),
                    "near_edges": view.get("near_edges"),
                    "quality": quality_details,
                }
            )
            if view.get("status") == "good_view_ready_for_graspnet":
                near_directory = view_directory
                state["events"].append(
                    {
                        "stage": "near_view_accepted",
                        "iteration": iteration,
                        "reason": "four-edge margin criteria",
                    }
                )
                break
            if (
                iteration >= arguments.max_refinements
                or view.get("status") == "best_available_view_ready_for_graspnet"
            ):
                _, near_directory, best_quality = max(
                    usable_near_views, key=lambda item: item[0]
                )
                best_source_directory = near_directory
                state["events"].append(
                    {
                        "stage": "best_available_near_view_selected",
                        "iteration": iteration,
                        "reason": (
                            "refinement limit reached"
                            if iteration >= arguments.max_refinements
                            else view.get("refinement_stop_reason")
                        ),
                        "quality": best_quality,
                    }
                )
                if best_source_directory != view_directory:
                    return_plan = run_child(
                        [
                            python,
                            str(repo / "scripts/plan_return_to_capture.py"),
                            str(best_source_directory),
                            "--output-root",
                            str(captures),
                            "--robotdata",
                            str(arguments.robotdata),
                            "--api-host",
                            "127.0.0.1",
                        ],
                        cwd=repo,
                        log=session / "06_best_view_return_plan.log",
                    )
                    best_return_plan = resolve_output(repo, str(return_plan["plan"]))
                    run_child(
                        [
                            python,
                            str(repo / "scripts/execute_observation_plan.py"),
                            str(best_return_plan),
                            "--robotdata",
                            str(arguments.robotdata),
                            "--execute-approved-observation",
                        ],
                        cwd=repo,
                        log=session / "07_best_view_return_execute.log",
                    )
                    try:
                        confirmed_view = run_child(
                            [*common_capture, "--capture-only"],
                            cwd=repo,
                            log=session / "07_best_view_recapture.log",
                        )
                    except RuntimeError:
                        state["events"].append(
                            {
                                "stage": "best_view_recapture_retry",
                                "reason": "first capture process failed",
                            }
                        )
                        confirmed_view = run_child(
                            [*common_capture, "--capture-only"],
                            cwd=repo,
                            log=session / "07_best_view_recapture_retry.log",
                        )
                    require(
                        confirmed_view.get("ok") is True,
                        "failed to recapture at best near-view pose",
                    )
                    near_directory = resolve_output(repo, str(confirmed_view["output"]))
                    _, confirmed_quality = near_view_quality(
                        near_directory, confirmed_view
                    )
                    state["events"].append(
                        {
                            "stage": "best_available_near_view_restored_and_recaptured",
                            "source": str(best_source_directory),
                            "confirmed": str(near_directory),
                            "quality": confirmed_quality,
                        }
                    )
                break
            require(view.get("status") == "awaiting_human_review", "no safe refinement")
            run_child(
                [
                    python,
                    str(repo / "scripts/execute_observation_plan.py"),
                    str(view_directory / "plan.json"),
                    "--robotdata",
                    str(arguments.robotdata),
                    "--execute-approved-observation",
                ],
                cwd=repo,
                log=session / f"07_refinement_{iteration}_execute.log",
            )
        require(near_directory is not None, "no usable near view")

        sys.path.insert(0, str(arguments.robotdata))
        from utils.arm_environment import DualArmEnvironment  # type: ignore
        from utils.kinematic_transforms import GripperKinematics  # type: ignore

        config = json.loads(
            (arguments.robotdata / "config/env_config.json").read_text("utf-8")
        )
        left_config = config["left_arm_config"]
        kinematics = GripperKinematics(
            left_config["base_extrinsic"]["R"],
            left_config["base_extrinsic"]["t"],
            left_config["gripper_extrinsic"]["R"],
            left_config["gripper_extrinsic"]["t"],
        )

        planning_attempts = []
        accepted = []
        last_strict = []
        for attempt in range(arguments.grasp_planning_attempts):
            log_name = (
                "08_graspnet_and_simulation.log"
                if attempt == 0
                else f"08_graspnet_and_simulation_retry_{attempt}.log"
            )
            evaluation_command = [
                    python,
                    str(repo / "scripts/evaluate_pistol_barrel_grasps.py"),
                    str(near_directory),
                    "--robotdata",
                    str(arguments.robotdata),
                ]
            if arguments.whole_object_grasp:
                evaluation_command.append("--whole-object")
            if arguments.fast_magazine:
                evaluation_command.append("--fast-magazine")
            orange_box_mode = "orange" in arguments.target_text.casefold()
            minimum_table_clearance_m = (
                0.007 if orange_box_mode else MINIMUM_TABLE_CLEARANCE_M
            )
            if orange_box_mode:
                evaluation_command.extend(
                    ["--minimum-table-clearance-m", str(minimum_table_clearance_m)]
                )
            run_child(
                evaluation_command,
                cwd=repo,
                log=session / log_name,
            )
            evaluation = json.loads(
                (near_directory / "barrel_grasp_evaluation.json").read_text("utf-8")
            )
            strict = strict_candidates(
                near_directory,
                evaluation,
                kinematics,
                relaxed_quality=arguments.relaxed_grasp_quality,
                regular_prism_mode=arguments.whole_object_grasp,
                fast_mode=arguments.fast_magazine,
                minimum_table_clearance_m=minimum_table_clearance_m,
            )
            last_strict = strict
            attempt_accepted = []
            for item in strict:
                if not item["accepted"]:
                    continue
                candidate = dict(item)
                candidate["planning_attempt"] = attempt
                attempt_accepted.append(candidate)
                accepted.append(candidate)
            attempt_validation = session / f"strict_grasp_validation_attempt_{attempt}.json"
            attempt_validation.write_text(
                json.dumps(
                    {
                        "near_capture": str(near_directory),
                        "quality_policy": state["grasp_quality_policy"],
                        "planning_attempt": attempt,
                        "candidates": strict,
                        "accepted_ranks": [item["rank"] for item in attempt_accepted],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            planning_attempts.append(
                {
                    "attempt": attempt,
                    "validation": str(attempt_validation),
                    "accepted_ranks": [item["rank"] for item in attempt_accepted],
                }
            )

        require(accepted, "no grasp passed quality and hard-safety validation")
        near_plan = json.loads((near_directory / "plan.json").read_text("utf-8"))
        lift_gateway = LegacyApiGateway("127.0.0.1", timeout_s=120.0)
        downstream_planning_attempts = []
        selected = None
        path = None
        expected_path_sha256 = None
        lift_plan = None
        box_transfer_plan = None
        selected_hardware_path_source = None
        seen_paths = set()
        for candidate in sorted(
            accepted, key=lambda item: item["selection_score"], reverse=True
        ):
            candidate_path = candidate["simulation_waypoints"]
            candidate_path_sha256 = candidate["simulation_path_sha256"]
            if candidate_path_sha256 in seen_paths:
                continue
            seen_paths.add(candidate_path_sha256)
            attempt_record = {
                "planning_attempt": candidate["planning_attempt"],
                "rank": candidate["rank"],
                "grasp_path_sha256": candidate_path_sha256,
            }
            try:
                require(
                    path_sha256(candidate_path) == candidate_path_sha256,
                    "simulation waypoint digest changed before downstream planning",
                )
                attempt_validation_path = Path(
                    planning_attempts[candidate["planning_attempt"]]["validation"]
                )
                attempt_validation = json.loads(
                    attempt_validation_path.read_text("utf-8")
                )
                simulated_candidate = next(
                    item
                    for item in attempt_validation["candidates"]
                    if int(item["rank"]) == int(candidate["rank"])
                )
                require(
                    bool(simulated_candidate["accepted"]),
                    "selected planning-attempt artifact no longer accepts the candidate",
                )
                require(
                    np.allclose(
                        np.asarray(candidate_path, dtype=float),
                        np.asarray(
                            simulated_candidate["simulation_waypoints"], dtype=float
                        ),
                        rtol=0.0,
                        atol=1e-9,
                    ),
                    "hardware path is not exactly the simulated path",
                )
                preview_directory = session / (
                    f"downstream_attempt_{candidate['planning_attempt']}_"
                    f"rank_{candidate['rank']}"
                )
                candidate_lift_plan = plan_post_grasp_lift(
                    lift_gateway,
                    kinematics,
                    candidate_path,
                    near_plan["right_joints"],
                    None if arguments.fast_magazine else preview_directory,
                    fast_mode=arguments.fast_magazine,
                )
                candidate_box_transfer_plan = plan_box_transfer(
                    lift_gateway,
                    kinematics,
                    candidate_lift_plan["path"][-1],
                    near_plan["right_joints"],
                    None if arguments.fast_magazine else preview_directory,
                    regular_prism_mode=arguments.whole_object_grasp,
                    fast_mode=arguments.fast_magazine,
                    reorient_before_translate=(
                        "orange" in arguments.target_text.casefold()
                    ),
                )
            except (RuntimeError, KeyError, StopIteration, ValueError) as error:
                attempt_record.update(
                    {"complete_workflow_feasible": False, "error": repr(error)}
                )
                downstream_planning_attempts.append(attempt_record)
                continue
            attempt_record.update(
                {
                    "complete_workflow_feasible": True,
                    "lift_path_sha256": candidate_lift_plan["path_sha256"],
                    "box_pre_release_path_sha256": candidate_box_transfer_plan[
                        "pre_release_path_sha256"
                    ],
                    "box_retreat_path_sha256": candidate_box_transfer_plan[
                        "retreat_path_sha256"
                    ],
                }
            )
            downstream_planning_attempts.append(attempt_record)
            selected = candidate
            path = candidate_path
            expected_path_sha256 = candidate_path_sha256
            lift_plan = candidate_lift_plan
            box_transfer_plan = candidate_box_transfer_plan
            selected_hardware_path_source = str(attempt_validation_path)
            break

        (session / "strict_grasp_validation.json").write_text(
            json.dumps(
                {
                    "near_capture": str(near_directory),
                    "quality_policy": state["grasp_quality_policy"],
                    "planning_attempts_requested": arguments.grasp_planning_attempts,
                    "planning_attempts": planning_attempts,
                    "downstream_planning_attempts": downstream_planning_attempts,
                    "candidates": last_strict,
                    "selected_attempt": (
                        selected["planning_attempt"] if selected is not None else None
                    ),
                    "selected_rank": selected["rank"] if selected is not None else None,
                    "selected_candidate": selected,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        state["strict_validation"] = str(session / "strict_grasp_validation.json")
        state["downstream_planning_attempts"] = downstream_planning_attempts
        require(
            selected is not None,
            "no validated grasp has a complete lift-and-box-transfer plan",
        )
        assert selected is not None
        assert path is not None
        assert expected_path_sha256 is not None
        assert lift_plan is not None
        assert box_transfer_plan is not None
        assert selected_hardware_path_source is not None
        state["selected_planning_attempt"] = selected["planning_attempt"]
        state["selected_rank"] = selected["rank"]
        state["events"].append(
            {
                "stage": "strict_grasp_selection",
                "planning_attempt": selected["planning_attempt"],
                "rank": selected["rank"],
                "cross_section_width_mm": selected["cross_section_width_mm"],
                "worst_side_clearance_mm": selected["worst_side_clearance_mm"],
                "smallest_joint_margin_deg": selected["smallest_joint_margin_deg"],
                "selected_advance_mm": selected["selected_advance_m"] * 1000.0,
                "minimum_path_virtual_tip_clearance_mm": selected[
                    "minimum_path_virtual_tip_clearance_mm"
                ],
                "simulation_path_sha256": selected["simulation_path_sha256"],
            }
        )

        require(
            path_sha256(path) == expected_path_sha256,
            "simulation waypoint digest changed before hardware execution",
        )
        state["hardware_path_source"] = selected_hardware_path_source
        state["hardware_path_sha256"] = expected_path_sha256
        state["hardware_path_waypoint_count"] = len(path)
        state["post_grasp_lift_plan"] = lift_plan
        state["events"].append(
            {
                "stage": "post_grasp_lift_simulated",
                "distance_mm": POST_GRASP_LIFT_M * 1000.0,
                "waypoint_count": lift_plan["waypoint_count"],
                "smallest_joint_margin_deg": lift_plan["smallest_joint_margin_deg"],
                "maximum_lateral_deviation_mm": lift_plan[
                    "maximum_lateral_deviation_mm"
                ],
                "path_sha256": lift_plan["path_sha256"],
            }
        )
        state["box_transfer_plan"] = box_transfer_plan
        state["events"].append(
            {
                "stage": "green_box_transfer_simulated",
                "drop_xy_m": box_transfer_plan["box"]["drop_xy_m"],
                "rim_z_m": box_transfer_plan["box"]["rim_z_m"],
                "requested_travel_z_m": box_transfer_plan["requested_travel_z_m"],
                "achieved_travel_z_m": box_transfer_plan["achieved_travel_z_m"],
                "minimum_safe_transfer_z_m": box_transfer_plan[
                    "minimum_safe_transfer_z_m"
                ],
                "extra_lift_ceiling": box_transfer_plan["extra_lift_ceiling"],
                "high_object_over_rim_clearance_mm": (
                    box_transfer_plan["high_object_over_rim_clearance_m"]
                    * 1000.0
                ),
                "release_gripper_over_rim_mm": (
                    box_transfer_plan["release_gripper_over_rim_m"] * 1000.0
                ),
                "smallest_joint_margin_deg": box_transfer_plan[
                    "smallest_joint_margin_deg"
                ],
                "pre_release_path_sha256": box_transfer_plan[
                    "pre_release_path_sha256"
                ],
                "retreat_path_sha256": box_transfer_plan["retreat_path_sha256"],
            }
        )
        environment = DualArmEnvironment(
            config["left_arm_config"],
            config["right_arm_config"],
            config["head_camera_config"],
        )
        selected_arm = environment.arm_left
        status, current = selected_arm.get_joint_degree()
        require(status == 0, "cannot read pre-execution joints")
        start_drift = float(
            np.max(np.abs(np.asarray(current) - np.asarray(near_plan["left_joints"])))
        )
        require(start_drift <= 0.5, f"stale grasp start state: {start_drift:.3f} deg")
        planned_start_drift = float(
            np.max(
                np.abs(
                    np.asarray(path[0], dtype=float)
                    - np.asarray(near_plan["left_joints"], dtype=float)
                )
            )
        )
        require(
            planned_start_drift <= 0.1,
            f"simulated path start differs from captured state: {planned_start_drift:.3f} deg",
        )
        actual_path_start_drift = float(
            np.max(np.abs(np.asarray(current, dtype=float) - np.asarray(path[0], dtype=float)))
        )
        require(
            actual_path_start_drift <= 0.5,
            f"hardware state differs from simulated path start: {actual_path_start_drift:.3f} deg",
        )
        state["pre_execution_start_drift_deg"] = start_drift
        state["simulated_path_start_drift_deg"] = planned_start_drift
        state["hardware_path_start_drift_deg"] = actual_path_start_drift

        hardware_video = HardwareVideoRecorder(
            {
                "left": environment.arm_left.camera,
                "right": environment.arm_right.camera,
                "head": environment.head_camera,
            },
            session,
            fps=10.0,
        )
        hardware_video.start()
        hardware_video.wait_for_required(("left", "right"), timeout_s=10.0)
        state["hardware_video"] = hardware_video.metadata()

        hardware_video.mark("gripper_opening")
        open_return = selected_arm.robot.rm_set_gripper_release(
            200, block=True, timeout=5
        )
        require(open_return == 0, f"open gripper failed: {open_return}")
        state["gripper_open_sent"] = True

        pregrasp_records = []
        hardware_video.mark("pregrasp_motion")
        for index, waypoint in enumerate(path[1:-1], start=1):
            return_code = selected_arm.robot.rm_movej(
                waypoint, v=5, r=0, connect=0, block=1
            )
            state["real_robot_motion_sent"] = True
            query_return, actual = selected_arm.get_joint_degree()
            error = float(np.max(np.abs(np.asarray(actual) - np.asarray(waypoint))))
            pregrasp_records.append(
                {"index": index, "return": return_code, "error_deg": error}
            )
            require(
                return_code == 0 and query_return == 0 and error <= 1.0,
                f"pregrasp waypoint {index} failed",
            )
        state["pregrasp_records"] = pregrasp_records
        camera_errors = save_recorded_observation(hardware_video, session, "pregrasp")
        for camera_name, camera_error in camera_errors.items():
            state["events"].append(
                {
                    "stage": "video_camera_unavailable",
                    "camera": camera_name,
                    "at": "pregrasp",
                    "error": camera_error,
                }
            )

        hardware_motion_speed = 4 if arguments.fast_magazine else 2
        hardware_video.mark("final_approach_motion")
        final_return = selected_arm.robot.rm_movej(
            path[-1], v=hardware_motion_speed, r=0, connect=0, block=1
        )
        state["real_robot_motion_sent"] = True
        query_return, final_joints = selected_arm.get_joint_degree()
        final_error = float(
            np.max(np.abs(np.asarray(final_joints) - np.asarray(path[-1])))
        )
        require(
            final_return == 0 and query_return == 0 and final_error <= 1.0,
            "final approach failed",
        )
        state["final_approach_error_deg"] = final_error
        camera_errors = save_recorded_observation(hardware_video, session, "approach")
        for camera_name, camera_error in camera_errors.items():
            state["events"].append(
                {
                    "stage": "video_camera_unavailable",
                    "camera": camera_name,
                    "at": "approach",
                    "error": camera_error,
                }
            )

        hardware_video.mark("gripper_closing")
        close_return = selected_arm.robot.rm_set_gripper_pick(
            arguments.close_speed,
            force=arguments.close_force,
            block=True,
            timeout=5,
        )
        state["gripper_close_sent"] = True
        gripper_return, gripper_state = selected_arm.robot.rm_get_gripper_state()
        state["close_return"] = close_return
        state["gripper_state"] = gripper_state
        require(close_return == 0 and gripper_return == 0, "gripper close failed")
        closed_wrist_target = None
        if arguments.fast_magazine:
            state["events"].append(
                {"stage": "magazine_post_close_visual_check_skipped"}
            )
        else:
            camera_errors = save_recorded_observation(hardware_video, session, "closed")
            for camera_name, camera_error in camera_errors.items():
                state["events"].append(
                    {
                        "stage": "video_camera_unavailable",
                        "camera": camera_name,
                        "at": "closed",
                        "error": camera_error,
                    }
                )
            closed_wrist_target = measure_wrist_target(
                lift_gateway,
                session / "closed_left_rgb.jpg",
                session / "closed_left_depth.npy",
                text=arguments.target_text,
            )
            state["closed_wrist_target"] = closed_wrist_target

        lift_path = lift_plan["path"]
        require(
            path_sha256(lift_path) == lift_plan["path_sha256"],
            "post-grasp lift path digest changed before hardware execution",
        )
        lift_start_status, lift_start_joints = selected_arm.get_joint_degree()
        lift_start_error = float(
            np.max(
                np.abs(
                    np.asarray(lift_start_joints, dtype=float)
                    - np.asarray(lift_path[0], dtype=float)
                )
            )
        )
        state["lift_path_start_error_deg"] = lift_start_error
        require(
            lift_start_status == 0 and lift_start_error <= 0.5,
            f"hardware state differs from lift path start: {lift_start_error:.3f} deg",
        )

        lift_records = []
        hardware_video.mark("post_grasp_vertical_lift")
        for index, waypoint in enumerate(lift_path[1:], start=1):
            return_code = selected_arm.robot.rm_movej(
                waypoint, v=hardware_motion_speed, r=0, connect=0, block=1
            )
            state["real_robot_motion_sent"] = True
            state["lift_sent"] = True
            query_return, actual = selected_arm.get_joint_degree()
            error = float(np.max(np.abs(np.asarray(actual) - np.asarray(waypoint))))
            lift_records.append(
                {"index": index, "return": return_code, "error_deg": error}
            )
            state["lift_records"] = lift_records
            require(
                return_code == 0 and query_return == 0 and error <= 1.0,
                f"post-grasp lift waypoint {index} failed",
            )

        lifted_status, lifted_joints = selected_arm.get_joint_degree()
        lifted_error = float(
            np.max(
                np.abs(
                    np.asarray(lifted_joints, dtype=float)
                    - np.asarray(lift_path[-1], dtype=float)
                )
            )
        )
        state["lift_final_error_deg"] = lifted_error
        require(
            lifted_status == 0 and lifted_error <= 0.5,
            f"post-grasp lift did not reach its endpoint: {lifted_error:.3f} deg",
        )
        state["lift_completed"] = True

        lifted_gripper_return, lifted_gripper_state = (
            selected_arm.robot.rm_get_gripper_state()
        )
        state["lifted_gripper_state"] = lifted_gripper_state
        require(lifted_gripper_return == 0, "cannot read gripper after vertical lift")
        expected_held_width_m = float(selected["cross_section_width_mm"]) / 1000.0
        initial_hold_validation = (
            {
                **validate_magazine_hold(lifted_gripper_state),
                "passed": float(lifted_gripper_state.get("actpos", -1.0)) >= 100.0,
                "policy": "orange_aperture_plus_visual",
            }
            if arguments.fast_magazine and is_orange_box
            else validate_magazine_hold(lifted_gripper_state)
            if arguments.fast_magazine
            else validate_held_object_aperture(
                lifted_gripper_state, expected_held_width_m
            )
        )
        state["post_lift_hold_validation"] = initial_hold_validation
        require(
            bool(initial_hold_validation["passed"]),
            "object retention check failed after initial lift",
        )
        if arguments.fast_magazine:
            state["events"].append(
                {"stage": "magazine_post_lift_visual_check_skipped"}
            )
        else:
            camera_errors = save_recorded_observation(
                hardware_video, session, "lifted"
            )
            for camera_name, camera_error in camera_errors.items():
                state["events"].append(
                    {
                        "stage": "video_camera_unavailable",
                        "camera": camera_name,
                        "at": "lifted",
                        "error": camera_error,
                    }
                )
            lifted_wrist_target = measure_wrist_target(
                lift_gateway,
                session / "lifted_left_rgb.jpg",
                session / "lifted_left_depth.npy",
                text=arguments.target_text,
            )
            initial_visual_retention = validate_wrist_target_retention(
                closed_wrist_target, lifted_wrist_target
            )
            state["post_lift_visual_retention"] = initial_visual_retention
            require(bool(initial_visual_retention["passed"]), "wrist-camera retention check failed after initial lift")

        pre_release_path = box_transfer_plan["pre_release_path"]
        retreat_path = box_transfer_plan["retreat_path"]
        require(
            path_sha256(pre_release_path)
            == box_transfer_plan["pre_release_path_sha256"],
            "green-box pre-release path digest changed before hardware execution",
        )
        require(
            path_sha256(retreat_path) == box_transfer_plan["retreat_path_sha256"],
            "green-box retreat path digest changed before hardware execution",
        )
        transfer_start_status, transfer_start_joints = selected_arm.get_joint_degree()
        transfer_start_error = float(
            np.max(
                np.abs(
                    np.asarray(transfer_start_joints, dtype=float)
                    - np.asarray(pre_release_path[0], dtype=float)
                )
            )
        )
        state["box_transfer_start_error_deg"] = transfer_start_error
        require(
            transfer_start_status == 0 and transfer_start_error <= 0.5,
            f"hardware state differs from box-transfer start: {transfer_start_error:.3f} deg",
        )

        transfer_records = []
        for segment in box_transfer_plan["pre_release_segments"]:
            phase = str(segment["phase"])
            segment_path = segment["path"]
            require(
                path_sha256(segment_path) == segment["path_sha256"],
                f"green-box {phase} path digest changed",
            )
            hardware_video.mark(f"green_box_{phase}")
            segment_status, segment_current = selected_arm.get_joint_degree()
            segment_start_error = float(
                np.max(
                    np.abs(
                        np.asarray(segment_current, dtype=float)
                        - np.asarray(segment_path[0], dtype=float)
                    )
                )
            )
            require(
                segment_status == 0 and segment_start_error <= 0.5,
                f"hardware state differs from green-box {phase} start",
            )
            for waypoint_index, waypoint in enumerate(segment_path[1:], start=1):
                return_code = selected_arm.robot.rm_movej(
                    waypoint, v=hardware_motion_speed, r=0, connect=0, block=1
                )
                state["real_robot_motion_sent"] = True
                state["box_transfer_sent"] = True
                query_return, actual = selected_arm.get_joint_degree()
                error = float(
                    np.max(np.abs(np.asarray(actual) - np.asarray(waypoint)))
                )
                transfer_records.append(
                    {
                        "phase": phase,
                        "index": waypoint_index,
                        "return": return_code,
                        "error_deg": error,
                    }
                )
                state["box_transfer_records"] = transfer_records
                require(
                    return_code == 0 and query_return == 0 and error <= 1.0,
                    f"green-box {phase} waypoint {waypoint_index} failed",
                )
            if phase.startswith("extra_lift_"):
                retention_return, retention_state = (
                    selected_arm.robot.rm_get_gripper_state()
                )
                require(
                    retention_return == 0,
                    f"cannot read gripper after green-box {phase}",
                )
                retention = (
                    {
                        **validate_magazine_hold(retention_state),
                        "passed": float(retention_state.get("actpos", -1.0)) >= 100.0,
                        "policy": "orange_aperture_plus_visual",
                    }
                    if arguments.fast_magazine and is_orange_box
                    else validate_magazine_hold(retention_state)
                    if arguments.fast_magazine
                    else validate_held_object_aperture(
                        retention_state, expected_held_width_m
                    )
                )
                state.setdefault("transfer_hold_validations", []).append(
                    {"phase": phase, **retention}
                )
                require(
                    bool(retention["passed"]),
                    f"object retention check failed after {phase}",
                )
                if arguments.fast_magazine:
                    continue
                prefix = f"retention_{phase}"
                camera_errors = save_recorded_observation(
                    hardware_video, session, prefix
                )
                for camera_name, camera_error in camera_errors.items():
                    state["events"].append(
                        {
                            "stage": "video_camera_unavailable",
                            "camera": camera_name,
                            "at": prefix,
                            "error": camera_error,
                        }
                    )
                current_wrist_target = measure_wrist_target(
                    lift_gateway,
                    session / f"{prefix}_left_rgb.jpg",
                    session / f"{prefix}_left_depth.npy",
                    text=arguments.target_text,
                )
                visual_retention = validate_wrist_target_retention(
                    closed_wrist_target, current_wrist_target
                )
                state.setdefault("transfer_visual_retentions", []).append(
                    {"phase": phase, **visual_retention}
                )
                require(
                    bool(visual_retention["passed"]),
                    (
                        f"wrist-camera retention check failed after {phase}: "
                        f"near target depth increased by "
                        f"{visual_retention['near_depth_increase_mm']:.1f} mm"
                    ),
                )

        if not arguments.fast_magazine:
            try:
                camera_errors = save_recorded_observation(
                    hardware_video, session, "green_box_release_ready"
                )
                for camera_name, camera_error in camera_errors.items():
                    state["events"].append(
                        {
                            "stage": "video_camera_unavailable",
                            "camera": camera_name,
                            "at": "green_box_release_ready",
                            "error": camera_error,
                        }
                    )
            except RuntimeError as capture_error:
                state["events"].append(
                    {
                        "stage": "green_box_release_capture_unavailable",
                        "error": repr(capture_error),
                    }
                )

        hardware_video.mark("green_box_release")
        release_return = selected_arm.robot.rm_set_gripper_release(
            200, block=True, timeout=5
        )
        state["box_release_sent"] = True
        state["box_release_return"] = release_return
        require(release_return == 0, f"green-box release failed: {release_return}")

        retreat_records = []
        for segment in box_transfer_plan["retreat_segments"]:
            phase = str(segment["phase"])
            segment_path = segment["path"]
            require(
                path_sha256(segment_path) == segment["path_sha256"],
                f"green-box {phase} path digest changed",
            )
            hardware_video.mark(f"green_box_{phase}")
            segment_status, segment_current = selected_arm.get_joint_degree()
            segment_start_error = float(
                np.max(
                    np.abs(
                        np.asarray(segment_current, dtype=float)
                        - np.asarray(segment_path[0], dtype=float)
                    )
                )
            )
            require(
                segment_status == 0 and segment_start_error <= 0.5,
                f"hardware state differs from green-box {phase} start",
            )
            for waypoint_index, waypoint in enumerate(segment_path[1:], start=1):
                return_code = selected_arm.robot.rm_movej(
                    waypoint, v=hardware_motion_speed, r=0, connect=0, block=1
                )
                state["real_robot_motion_sent"] = True
                query_return, actual = selected_arm.get_joint_degree()
                error = float(
                    np.max(np.abs(np.asarray(actual) - np.asarray(waypoint)))
                )
                retreat_records.append(
                    {
                        "phase": phase,
                        "index": waypoint_index,
                        "return": return_code,
                        "error_deg": error,
                    }
                )
                state["box_retreat_records"] = retreat_records
                require(
                    return_code == 0 and query_return == 0 and error <= 1.0,
                    f"green-box {phase} waypoint {waypoint_index} failed",
                )
        state["box_retreat_completed"] = True

        released_gripper_return, released_gripper_state = (
            selected_arm.robot.rm_get_gripper_state()
        )
        state["released_gripper_state"] = released_gripper_state
        require(released_gripper_return == 0, "cannot read gripper after box release")
        try:
            camera_errors = save_recorded_observation(
                hardware_video, session, "green_box_retreated"
            )
            for camera_name, camera_error in camera_errors.items():
                state["events"].append(
                    {
                        "stage": "video_camera_unavailable",
                        "camera": camera_name,
                        "at": "green_box_retreated",
                        "error": camera_error,
                    }
                )
        except RuntimeError as capture_error:
            state["events"].append(
                {
                    "stage": "green_box_retreat_capture_unavailable",
                    "error": repr(capture_error),
                }
            )

        state["hardware_video"] = hardware_video.stop()

        state["ok"] = True
        state["status"] = "placed_in_green_box_and_retreated_100mm"
        print(json.dumps(state, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        state["error"] = repr(error)
        state["traceback"] = traceback.format_exc()
        if selected_arm is not None:
            try:
                state["emergency_stop_return"] = selected_arm.robot.rm_set_arm_stop()
            except Exception as stop_error:
                state["emergency_stop_error"] = repr(stop_error)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        return 1
    finally:
        session.mkdir(parents=True, exist_ok=True)
        if hardware_video is not None:
            state["hardware_video"] = hardware_video.stop()
        (session / "result.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if environment is not None:
            if hasattr(environment, "head_camera"):
                environment.head_camera.close()
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
