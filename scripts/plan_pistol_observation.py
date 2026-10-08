#!/usr/bin/env python3
"""Capture the current pistol view and create one observation-motion preview.

The script is intentionally planning-only: it reads robot/camera state, calls
Florence, evaluates the detected box with ``ObservationRefiner`` policy, asks
the existing simulation API for a path/video, and never invokes a robot action.
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
from robot_workflow.observation_refinement import (
    AdjustmentKind,
    BoundingBox,
    BoxEdge,
    CameraAdjustment,
    ImageSize,
    ObservationRefinementConfig,
    plan_observation_adjustment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robotdata",
        type=Path,
        default=Path("RobotDataCollection"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("grasp_api_tests"),
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument(
        "--detection-camera",
        choices=("arm", "left", "right", "head"),
        default="arm",
    )
    parser.add_argument("--head-extrinsic-report", type=Path)
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="capture/detect/compute a point-cloud center without planning adjustment",
    )
    parser.add_argument("--text", default="pistol")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument(
        "--max-camera-z",
        type=float,
        default=0.40,
        help="absolute world-z ceiling for every planned wrist-camera refinement",
    )
    parser.add_argument(
        "--height-only-refinement",
        action="store_true",
        help="for a bad near view, try raising 3 cm with 2/1 cm fallbacks; disable translation/rotation",
    )
    return parser.parse_args()


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    rx_matrix = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_matrix = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_matrix = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz_matrix @ ry_matrix @ rx_matrix


def euler_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    ry = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(ry)) > 1e-6:
        rz = math.atan2(matrix[1, 0], matrix[0, 0])
        rx = math.atan2(matrix[2, 1], matrix[2, 2])
    else:
        rz = 0.0
        rx = math.atan2(matrix[0, 1], matrix[1, 1])
    return rx, ry, rz


def mask_bbox(mask: np.ndarray) -> BoundingBox:
    rows, columns = np.nonzero(mask)
    if not len(columns):
        raise RuntimeError("Florence returned an empty target mask")
    return BoundingBox(
        float(columns.min()),
        float(rows.min()),
        float(columns.max() + 1),
        float(rows.max() + 1),
    )


def normalize_mask(value: Any) -> np.ndarray:
    """Normalize SAM2's HxW or 1xHxW mask for one Florence box."""

    mask = np.asarray(value, dtype=bool)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise RuntimeError(f"unexpected per-detection mask shape: {mask.shape}")
    return mask


def orange_color_component_mask(image_rgb: np.ndarray) -> np.ndarray | None:
    """Extract the largest compact, saturated-orange component in one view."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    color = (
        (hsv[:, :, 0] >= 3)
        & (hsv[:, :, 0] <= 28)
        & (hsv[:, :, 1] >= 110)
        & (hsv[:, :, 2] >= 90)
    ).astype(np.uint8)
    color = cv2.morphologyEx(color, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, components, statistics, _ = cv2.connectedComponentsWithStats(
        color, connectivity=8
    )
    eligible = [
        index
        for index in range(1, count)
        if 200
        <= int(statistics[index, cv2.CC_STAT_AREA])
        <= int(0.25 * image_rgb.shape[0] * image_rgb.shape[1])
    ]
    if not eligible:
        return None
    selected = max(
        eligible, key=lambda index: int(statistics[index, cv2.CC_STAT_AREA])
    )
    return components == selected


def masked_world_center(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: dict[str, Any],
    camera_pose: list[float],
) -> tuple[list[float], int, list[float]]:
    """Return a robust world center from an eroded SAM mask and depth."""

    eroded = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    valid = eroded & np.isfinite(depth_mm) & (depth_mm > 0)
    values = depth_mm[valid].astype(float)
    if values.size < 200:
        raise RuntimeError(f"only {values.size} valid target depth pixels")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    tolerance = max(3.0 * 1.4826 * mad, 8.0)
    valid &= np.abs(depth_mm.astype(float) - median) <= tolerance
    rows, columns = np.nonzero(valid)
    depths_m = depth_mm[valid].astype(float) / 1000.0
    x = (columns - float(intrinsics["ppx"])) * depths_m / float(intrinsics["fx"])
    y = (rows - float(intrinsics["ppy"])) * depths_m / float(intrinsics["fy"])
    points_camera = np.column_stack((x, y, depths_m))
    camera_rotation = rotation_matrix(*camera_pose[3:])
    points_world = points_camera @ camera_rotation.T + np.asarray(camera_pose[:3])
    center = np.median(points_world, axis=0)
    quantiles = np.quantile(depths_m * 1000.0, [0.1, 0.5, 0.9]).tolist()
    return center.tolist(), int(len(points_world)), [float(value) for value in quantiles]


def adjustment_dict(adjustment: CameraAdjustment) -> dict[str, Any]:
    return {
        "kind": adjustment.kind.value,
        "x_m": adjustment.x_m,
        "y_m": adjustment.y_m,
        "z_m": adjustment.z_m,
        "roll_rad": adjustment.roll_rad,
        "pitch_rad": adjustment.pitch_rad,
        "yaw_rad": adjustment.yaw_rad,
        "reason": adjustment.reason,
        "fallback_for": (
            adjustment.fallback_for.value if adjustment.fallback_for is not None else None
        ),
    }


def adaptive_translation(
    adjustment: CameraAdjustment,
    near_edges: tuple[BoxEdge, ...],
    box: BoundingBox,
    image_size: ImageSize,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict[str, Any],
    edge_margin_ratio: float,
    maximum_step_m: float,
) -> CameraAdjustment:
    """Scale lateral motion from the actual pixel deficit and scene depth."""

    if adjustment.kind is not AdjustmentKind.TRANSLATE:
        return adjustment
    valid_depth = depth_mm[mask & np.isfinite(depth_mm) & (depth_mm > 0)].astype(float)
    if valid_depth.size < 200:
        return adjustment
    depth_m = float(np.median(valid_depth)) / 1000.0
    horizontal_margin = image_size.width * edge_margin_ratio
    vertical_margin = image_size.height * edge_margin_ratio
    edge_set = set(near_edges)

    x_pixels = 0.0
    if BoxEdge.LEFT in edge_set:
        x_pixels -= max(0.0, horizontal_margin - box.x_min)
    if BoxEdge.RIGHT in edge_set:
        x_pixels += max(0.0, box.x_max - (image_size.width - horizontal_margin))
    y_pixels = 0.0
    if BoxEdge.TOP in edge_set:
        y_pixels -= max(0.0, vertical_margin - box.y_min)
    if BoxEdge.BOTTOM in edge_set:
        y_pixels += max(0.0, box.y_max - (image_size.height - vertical_margin))

    def convert(pixels: float, focal_length: float) -> float:
        if pixels == 0.0:
            return 0.0
        magnitude = 1.10 * abs(pixels) * depth_m / focal_length
        magnitude = min(maximum_step_m, max(0.003, magnitude))
        return math.copysign(magnitude, pixels)

    return CameraAdjustment(
        kind=AdjustmentKind.TRANSLATE,
        x_m=convert(x_pixels, float(intrinsics["fx"])),
        y_m=convert(y_pixels, float(intrinsics["fy"])),
        reason="adaptive translation from edge pixel deficit, depth, and focal length",
    )


def target_camera_pose(
    current_pose: list[float], adjustment: CameraAdjustment
) -> list[float]:
    current_rotation = rotation_matrix(*current_pose[3:])
    position = np.asarray(current_pose[:3], dtype=float)
    if adjustment.kind in (AdjustmentKind.TRANSLATE, AdjustmentKind.HEIGHT):
        position = (
            position
            + current_rotation[:, 0] * adjustment.x_m
            + current_rotation[:, 1] * adjustment.y_m
            + np.array([0.0, 0.0, adjustment.z_m])
        )
        target_rotation = current_rotation
    elif adjustment.kind is AdjustmentKind.ROTATE:
        target_rotation = current_rotation @ rotation_matrix(
            adjustment.roll_rad,
            adjustment.pitch_rad,
            adjustment.yaw_rad,
        )
    else:
        target_rotation = current_rotation
    return [*position.tolist(), *euler_from_matrix(target_rotation)]


def camera_pose_to_arm_end_pose(
    camera_pose: list[float], camera_extrinsic: dict[str, Any]
) -> list[float]:
    """Convert world-from-camera to world-from-Link7 without requiring IK."""

    world_from_camera_rotation = rotation_matrix(*camera_pose[3:])
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
        *euler_from_matrix(world_from_body_rotation),
    ]


def main() -> int:
    arguments = parse_args()
    if not 0.30 <= arguments.max_camera_z <= 0.40:
        raise RuntimeError("max camera z must be within 0.30..0.40 m")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = arguments.output_root / f"pistol_refine_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(arguments.robotdata))
    from utils.arm_environment import DualArmEnvironment  # type: ignore

    config_path = arguments.robotdata / "config" / "env_config.json"
    config = json.loads(config_path.read_text("utf-8"))
    environment = None
    try:
        environment = DualArmEnvironment(
            config["left_arm_config"],
            config["right_arm_config"],
            config["head_camera_config"],
        )
        observation, head_capture_error = capture_required_wrists_optional_head(
            environment
        )
        arm_data = observation[arguments.arm]
        arm_object = (
            environment.arm_left if arguments.arm == "left" else environment.arm_right
        )

        for name in ("left", "head", "right"):
            if name not in observation:
                continue
            rgb = np.asarray(observation[name]["rgb"])
            depth = np.asarray(observation[name]["depth"])
            cv2.imwrite(str(output / f"{name}_rgb.jpg"), rgb[:, :, ::-1])
            np.save(output / f"{name}_depth.npy", depth, allow_pickle=False)

        detection_camera = (
            arguments.arm
            if arguments.detection_camera == "arm"
            else arguments.detection_camera
        )
        detection_data = observation[detection_camera]
        image = np.asarray(detection_data["rgb"])
        gateway = LegacyApiGateway(arguments.api_host, timeout_s=120.0)
        detection = gateway.open_vocabulary(
            image_bytes=image.tobytes(),
            image_shape=image.shape,
            text=arguments.text,
        )
        masks = [normalize_mask(value) for value in detection["masks"]]
        if not masks:
            raise RuntimeError("Florence did not detect the pistol")
        selected_mask = max(masks, key=np.count_nonzero)
        segmentation_source = "florence_sam"
        if "orange" in arguments.text.casefold():
            color_mask = orange_color_component_mask(image)
            if color_mask is not None:
                selected_mask = color_mask
                segmentation_source = "orange_color_component"
        np.save(output / "target_mask.npy", selected_mask, allow_pickle=False)
        box = mask_bbox(selected_mask)
        size = ImageSize(image.shape[1], image.shape[0])
        policy = ObservationRefinementConfig(
            edge_margin_ratio=0.10,
            height_step_m=0.03,
            translation_step_m=0.03,
            rotation_step_rad=math.radians(8.0),
            max_iterations=5,
        )
        decision = plan_observation_adjustment(box, size, policy)
        decision = type(decision)(
            near_edges=decision.near_edges,
            primary=adaptive_translation(
                decision.primary,
                decision.near_edges,
                box,
                size,
                np.asarray(detection_data["depth"]),
                selected_mask,
                detection_data["intrinsics"],
                policy.edge_margin_ratio,
                policy.translation_step_m,
            ),
            height_fallbacks=decision.height_fallbacks,
            rotation_fallbacks=decision.rotation_fallbacks,
        )
        if arguments.height_only_refinement and not decision.good_view:
            decision = type(decision)(
                near_edges=decision.near_edges,
                primary=CameraAdjustment(
                    kind=AdjustmentKind.HEIGHT,
                    z_m=policy.height_step_m,
                    reason="height-only mode: raise camera by up to 3 cm",
                ),
                rotation_fallbacks=(),
            )

        overlay = image[:, :, ::-1].copy()
        cv2.rectangle(
            overlay,
            (round(box.x_min), round(box.y_min)),
            (round(box.x_max) - 1, round(box.y_max) - 1),
            (0, 255, 255),
            2,
        )
        label = "edges=" + ",".join(edge.value for edge in decision.near_edges)
        cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imwrite(str(output / "detection_and_edges.jpg"), overlay)

        current_joints = [float(value) for value in arm_data["joints"]]
        head_extrinsic_status = None
        if detection_camera == "head":
            if arguments.head_extrinsic_report is None:
                raise RuntimeError("head detection requires --head-extrinsic-report")
            head_report = json.loads(arguments.head_extrinsic_report.read_text("utf-8"))
            world_from_head = np.asarray(
                head_report["world_from_head_camera"], dtype=float
            )
            if world_from_head.shape != (4, 4):
                raise RuntimeError("invalid world_from_head_camera matrix")
            current_camera_pose = [
                *world_from_head[:3, 3].tolist(),
                *euler_from_matrix(world_from_head[:3, :3]),
            ]
            head_extrinsic_status = head_report.get("status")
        else:
            camera_arm = (
                environment.arm_left
                if detection_camera == "left"
                else environment.arm_right
            )
            camera_joints = [
                float(value) for value in observation[detection_camera]["joints"]
            ]
            current_camera_pose = [
                float(value)
                for value in camera_arm.camera_kinematics.get_camera_forward(camera_joints)
            ]
        center_world, center_point_count, depth_quantiles_mm = masked_world_center(
            selected_mask,
            np.asarray(detection_data["depth"]),
            detection_data["intrinsics"],
            current_camera_pose,
        )
        result: dict[str, Any] = {
            "ok": True,
            "planning_only": True,
            "real_robot_command_sent": False,
            "gripper_command_sent": False,
            "arm": arguments.arm,
            "detection_camera": detection_camera,
            "head_extrinsic_status": head_extrinsic_status,
            "head_camera_available": "head" in observation,
            "head_camera_error": head_capture_error,
            "text": arguments.text,
            "labels": list(detection["labels"]),
            "segmentation_source": segmentation_source,
            "bbox_xyxy": [box.x_min, box.y_min, box.x_max, box.y_max],
            "near_edges": [edge.value for edge in decision.near_edges],
            "current_camera_pose": current_camera_pose,
            "current_joints": current_joints,
            "left_joints": [float(value) for value in observation["left"]["joints"]],
            "right_joints": [float(value) for value in observation["right"]["joints"]],
            "sam_mask_world_center_m": center_world,
            "sam_mask_world_point_count": center_point_count,
            "sam_mask_depth_quantiles_mm": depth_quantiles_mm,
            "maximum_camera_z_m": float(arguments.max_camera_z),
            "primary": adjustment_dict(decision.primary),
            "attempts": [],
        }

        if arguments.capture_only:
            result["status"] = "capture_ready"
        elif decision.good_view:
            result["status"] = "good_view_ready_for_graspnet"
        else:
            candidates = [decision.primary, *decision.height_fallbacks]
            if decision.primary.kind is AdjustmentKind.HEIGHT:
                existing_heights = {
                    round(float(candidate.z_m), 6)
                    for candidate in candidates
                    if candidate.kind is AdjustmentKind.HEIGHT
                }
                for fallback_height_m in (0.02, 0.01):
                    if (
                        fallback_height_m < float(decision.primary.z_m)
                        and round(fallback_height_m, 6) not in existing_heights
                    ):
                        candidates.append(
                            CameraAdjustment(
                                kind=AdjustmentKind.HEIGHT,
                                z_m=fallback_height_m,
                                reason=(
                                    "fallback when the requested camera-height "
                                    "adjustment has no simulated solution"
                                ),
                                fallback_for=AdjustmentKind.HEIGHT,
                            )
                        )
            other_arm = "right" if arguments.arm == "left" else "left"
            arm_configuration = config[f"{arguments.arm}_arm_config"]
            for adjustment in candidates:
                candidate_pose = target_camera_pose(current_camera_pose, adjustment)
                target_joints = arm_object.camera_kinematics.get_camera_inverse(
                    current_joints, candidate_pose
                )
                if target_joints is not None:
                    target_joints = [float(value) for value in target_joints]
                target_end_pose = camera_pose_to_arm_end_pose(
                    candidate_pose, arm_configuration["cam_extrinsic"]
                )
                attempt: dict[str, Any] = {
                    "adjustment": adjustment_dict(adjustment),
                    "target_camera_pose": candidate_pose,
                    "local_ik_solved": target_joints is not None,
                    "target_pose_source": "camera pose plus fixed camera extrinsic",
                    "target_end_pose": target_end_pose,
                    "within_camera_height_limit": bool(
                        float(candidate_pose[2]) <= arguments.max_camera_z + 1e-9
                    ),
                }
                result["attempts"].append(attempt)
                if not attempt["within_camera_height_limit"]:
                    attempt["simulation_solved"] = False
                    attempt["rejection_reason"] = "absolute camera-height limit"
                    continue
                if target_joints is not None:
                    attempt["target_joints"] = target_joints
                simulation = gateway.simulate(
                    target_pose=target_end_pose,
                    arm=arguments.arm,
                    left_joints_deg=observation["left"]["joints"],
                    right_joints_deg=observation["right"]["joints"],
                    num_waypoints=10,
                    recording=True,
                )
                attempt["simulation_solved"] = bool(simulation["success"])
                if not simulation["success"]:
                    continue
                video_path = output / "observation_adjustment_preview.mp4"
                video_path.write_bytes(base64.b64decode(simulation["video"]))
                attempt["waypoints"] = simulation["path"]
                result["status"] = "awaiting_human_review"
                result["selected_adjustment"] = adjustment_dict(adjustment)
                result["selected_target_joints"] = target_joints
                result["selected_target_end_pose"] = target_end_pose
                result["preview_video"] = str(video_path)
                break
            else:
                result["status"] = "best_available_view_ready_for_graspnet"
                result["refinement_stop_reason"] = (
                    "no further camera adjustment passed the simulation and "
                    "absolute-height policy; keep this capture as a grasp-input "
                    "candidate"
                )

        (output / "plan.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps({"output": str(output), **result}, ensure_ascii=False))
        return 0 if result["ok"] else 2
    finally:
        if environment is not None:
            if hasattr(environment, "head_camera"):
                environment.head_camera.close()
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
