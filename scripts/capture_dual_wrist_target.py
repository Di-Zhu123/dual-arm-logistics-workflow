#!/usr/bin/env python3
"""Locate a target from the original side-by-side dual wrist RGB-D view."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robotdata",
        type=Path,
        default=Path("RobotDataCollection"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("grasp_api_tests"))
    parser.add_argument("--text", default="pistol")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument(
        "--max-mask-fraction",
        type=float,
        default=0.25,
        help="Reject detections covering more than this fraction of the stitched image.",
    )
    return parser.parse_args()


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


def masked_world_center(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: dict[str, Any],
    camera_pose: list[float],
) -> tuple[np.ndarray, int, list[float]]:
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
    points_camera = np.column_stack(
        (
            (columns - float(intrinsics["ppx"])) * depths_m / float(intrinsics["fx"]),
            (rows - float(intrinsics["ppy"])) * depths_m / float(intrinsics["fy"]),
            depths_m,
        )
    )
    points_world = points_camera @ rotation_matrix(*camera_pose[3:]).T + np.asarray(
        camera_pose[:3]
    )
    return (
        np.median(points_world, axis=0),
        int(len(points_world)),
        [float(value) for value in np.quantile(depths_m * 1000.0, [0.1, 0.5, 0.9])],
    )


def mask_bbox(mask: np.ndarray) -> list[float]:
    rows, columns = np.nonzero(mask)
    if not len(columns):
        raise RuntimeError("empty mask")
    return [
        float(columns.min()),
        float(rows.min()),
        float(columns.max() + 1),
        float(rows.max() + 1),
    ]


def normalize_mask(value: Any) -> np.ndarray:
    """Normalize SAM2's HxW or 1xHxW mask for one Florence box."""

    mask = np.asarray(value, dtype=bool)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise RuntimeError(f"unexpected per-detection mask shape: {mask.shape}")
    return mask


def orange_color_component_masks(
    stitched_rgb: np.ndarray, side_width: int
) -> list[np.ndarray]:
    """Return the largest saturated-orange component from each wrist image."""
    masks: list[np.ndarray] = []
    for offset in (0, side_width):
        side = stitched_rgb[:, offset : offset + side_width]
        hsv = cv2.cvtColor(side, cv2.COLOR_RGB2HSV)
        color = (
            (hsv[:, :, 0] >= 3)
            & (hsv[:, :, 0] <= 28)
            & (hsv[:, :, 1] >= 110)
            & (hsv[:, :, 2] >= 90)
        ).astype(np.uint8)
        color = cv2.morphologyEx(
            color, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        color = cv2.morphologyEx(
            color, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
        )
        count, components, statistics, _ = cv2.connectedComponentsWithStats(
            color, connectivity=8
        )
        eligible = [
            index
            for index in range(1, count)
            if 200
            <= int(statistics[index, cv2.CC_STAT_AREA])
            <= int(0.20 * side.shape[0] * side.shape[1])
        ]
        if not eligible:
            continue
        selected = max(
            eligible, key=lambda index: int(statistics[index, cv2.CC_STAT_AREA])
        )
        stitched_mask = np.zeros(stitched_rgb.shape[:2], dtype=bool)
        stitched_mask[:, offset : offset + side_width] = components == selected
        masks.append(stitched_mask)
    return masks


def main() -> int:
    arguments = parse_args()
    if not 0.0 < arguments.max_mask_fraction < 1.0:
        raise ValueError("--max-mask-fraction must be between 0 and 1")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = arguments.output_root / f"dual_wrist_target_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(arguments.robotdata))
    from utils.arm_environment import DualArmEnvironment  # type: ignore

    config = json.loads(
        (arguments.robotdata / "config/env_config.json").read_text("utf-8")
    )
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
        for name in ("left", "right", "head"):
            if name not in observation:
                continue
            rgb = np.asarray(observation[name]["rgb"])
            depth = np.asarray(observation[name]["depth"])
            cv2.imwrite(str(output / f"{name}_rgb.jpg"), rgb[:, :, ::-1])
            np.save(output / f"{name}_depth.npy", depth, allow_pickle=False)

        left_rgb = np.asarray(observation["left"]["rgb"])
        right_rgb = np.asarray(observation["right"]["rgb"])
        if left_rgb.shape != right_rgb.shape:
            raise RuntimeError("left/right wrist RGB dimensions differ")
        stitched = np.hstack((left_rgb, right_rgb))
        cv2.imwrite(str(output / "stitched_rgb.jpg"), stitched[:, :, ::-1])
        detection = LegacyApiGateway(arguments.api_host, timeout_s=120.0).open_vocabulary(
            image_bytes=stitched.tobytes(),
            image_shape=stitched.shape,
            text=arguments.text,
        )
        masks = [normalize_mask(value) for value in detection["masks"]]
        labels = [str(value) for value in detection["labels"]]
        if "orange" in arguments.text.casefold():
            color_masks = orange_color_component_masks(stitched, left_rgb.shape[1])
            masks.extend(color_masks)
            labels.extend(["orange_color_component"] * len(color_masks))
        if not masks:
            raise RuntimeError("Florence+SAM did not detect the target in either wrist view")

        width = left_rgb.shape[1]
        candidates: list[dict[str, Any]] = []
        rejected_detections: list[dict[str, Any]] = []
        camera_poses: dict[str, list[float]] = {}
        for name, arm in (("left", environment.arm_left), ("right", environment.arm_right)):
            camera_poses[name] = [
                float(value)
                for value in arm.camera_kinematics.get_camera_forward(
                    observation[name]["joints"]
                )
            ]
        for index, mask in enumerate(masks):
            if mask.shape != stitched.shape[:2]:
                raise RuntimeError("detector returned a mask with an unexpected shape")
            label = labels[index] if index < len(labels) else arguments.text
            mask_pixels = int(np.count_nonzero(mask))
            mask_fraction = mask_pixels / float(mask.size)
            if mask_fraction > arguments.max_mask_fraction:
                rejected_detections.append(
                    {
                        "mask_index": index,
                        "label": label,
                        "reason": "mask_too_large",
                        "mask_pixels": mask_pixels,
                        "mask_fraction": mask_fraction,
                        "bbox_xyxy": mask_bbox(mask),
                    }
                )
                continue
            for name, offset in (("left", 0), ("right", width)):
                side_mask = mask[:, offset : offset + width]
                if int(np.count_nonzero(side_mask)) < 200:
                    continue
                try:
                    center, count, depth_quantiles = masked_world_center(
                        side_mask,
                        np.asarray(observation[name]["depth"]),
                        observation[name]["intrinsics"],
                        camera_poses[name],
                    )
                except RuntimeError:
                    continue
                candidates.append(
                    {
                        "mask_index": index,
                        "camera": name,
                        "label": label,
                        "center": center,
                        "point_count": count,
                        "depth_quantiles_mm": depth_quantiles,
                        "mask_fraction": mask_fraction,
                        "side_mask": side_mask,
                        "stitched_mask": mask,
                    }
                )
        if not candidates:
            rejection_summary = "; ".join(
                f"{item['label']}: {item['reason']} "
                f"({100.0 * item['mask_fraction']:.1f}%)"
                for item in rejected_detections
            )
            detail = f"; rejected detections: {rejection_summary}" if rejection_summary else ""
            raise RuntimeError(f"SAM masks contained no usable wrist-camera depth{detail}")

        pair = None
        pair_score = -1
        for left in [item for item in candidates if item["camera"] == "left"]:
            for right in [item for item in candidates if item["camera"] == "right"]:
                if left["label"].casefold() != right["label"].casefold():
                    continue
                distance = float(np.linalg.norm(left["center"] - right["center"]))
                if distance > 0.08:
                    continue
                score = int(left["point_count"]) + int(right["point_count"])
                if score > pair_score:
                    pair = (left, right, distance)
                    pair_score = score

        if pair is not None:
            selected = max(pair[:2], key=lambda item: int(item["point_count"]))
            center_world = np.median(
                np.vstack((pair[0]["center"], pair[1]["center"])), axis=0
            )
            fusion = {
                "mode": "matched_left_right",
                "distance_mm": pair[2] * 1000.0,
                "members": [pair[0]["camera"], pair[1]["camera"]],
            }
        else:
            selected = max(candidates, key=lambda item: int(item["point_count"]))
            center_world = selected["center"]
            fusion = {"mode": "single_best_depth_mask", "members": [selected["camera"]]}

        np.save(output / "target_mask.npy", selected["stitched_mask"], allow_pickle=False)
        np.save(output / "selected_wrist_mask.npy", selected["side_mask"], allow_pickle=False)
        overlay = stitched[:, :, ::-1].copy()
        overlay[selected["stitched_mask"]] = (
            0.55 * overlay[selected["stitched_mask"]] + 0.45 * np.array([0, 255, 0])
        ).astype(np.uint8)
        cv2.imwrite(str(output / "stitched_target_mask.jpg"), overlay)

        selected_arm = arguments.arm
        result = {
            "ok": True,
            "planning_only": True,
            "real_robot_command_sent": False,
            "gripper_command_sent": False,
            "status": "capture_ready",
            "arm": selected_arm,
            "detection_camera": "dual_wrist_stitched",
            "text": arguments.text,
            "labels": labels,
            "bbox_xyxy": mask_bbox(selected["stitched_mask"]),
            "selected_mask_fraction": selected["mask_fraction"],
            "max_mask_fraction": arguments.max_mask_fraction,
            "rejected_detections": rejected_detections,
            "selected_depth_camera": selected["camera"],
            "fusion": fusion,
            "sam_mask_world_center_m": [float(value) for value in center_world],
            "sam_mask_world_point_count": int(pair_score if pair is not None else selected["point_count"]),
            "sam_mask_depth_quantiles_mm": selected["depth_quantiles_mm"],
            "current_joints": [float(value) for value in observation[selected_arm]["joints"]],
            "left_joints": [float(value) for value in observation["left"]["joints"]],
            "right_joints": [float(value) for value in observation["right"]["joints"]],
            "current_camera_pose": camera_poses[selected_arm],
            "head_used_for_geometry": False,
            "head_camera_available": "head" in observation,
            "head_camera_error": head_capture_error,
        }
        plan_path = output / "plan.json"
        plan_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
        print(json.dumps({"output": str(output), "plan": str(plan_path), **result}, ensure_ascii=False))
        return 0
    finally:
        if environment is not None:
            if hasattr(environment, "head_camera"):
                environment.head_camera.close()
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
