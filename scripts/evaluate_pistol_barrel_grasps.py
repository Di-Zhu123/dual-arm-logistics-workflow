#!/usr/bin/env python3
"""Segment a pistol barrel from a near RGB-D view and validate GraspNet output.

This program has no robot-execution path.  It saves masks, candidate overlays,
JSON diagnostics and (when possible) simulation videos for human review.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

from robot_workflow.gripper_clearance import (
    FINAL_INSERTION_BIAS_M,
    KINEMATIC_TCP_LENGTH_M,
    MINIMUM_TABLE_CLEARANCE_M,
    SIMULATED_OPEN_MESH_LENGTH_M,
    VIRTUAL_COLLISION_LENGTH_M,
    candidate_advances_m,
    virtual_tip_clearance_m,
)
from robot_workflow.legacy_tcp import LegacyApiGateway


MAX_GRIPPER_WIDTH_M = 0.0654
LEFT_ARM_BASE_WORLD_M = np.array([0.0, 0.25, 0.0])
PREGRASP_RETRACTION_M = 0.05


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ]
    )


def mask_bbox(mask: np.ndarray) -> list[int]:
    rows, columns = np.nonzero(mask)
    if not len(columns):
        raise RuntimeError("empty mask")
    return [int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)]


def world_center(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: dict[str, float],
    camera_pose: list[float],
) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    valid = eroded & (depth_mm > 0)
    depths = depth_mm[valid].astype(float)
    if depths.size < 500:
        raise RuntimeError(f"only {depths.size} valid barrel pixels")
    median = float(np.median(depths))
    mad = float(np.median(np.abs(depths - median)))
    tolerance = max(3.0 * 1.4826 * mad, 8.0)
    valid &= np.abs(depth_mm.astype(float) - median) <= tolerance
    rows, columns = np.nonzero(valid)
    z = depth_mm[valid].astype(float) / 1000.0
    x = (columns - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (rows - intrinsics["ppy"]) * z / intrinsics["fy"]
    points_camera = np.column_stack((x, y, z))
    rotation = rotation_matrix(*camera_pose[3:])
    points_world = points_camera @ rotation.T + np.asarray(camera_pose[:3])
    center = np.median(points_world, axis=0)
    covariance = np.cov(points_world[:, :2], rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    minor_axis = eigenvectors[:, 0]
    preferred_yaw_degrees = math.degrees(
        math.atan2(-float(minor_axis[0]), float(minor_axis[1]))
    )
    preferred_yaw_degrees = (preferred_yaw_degrees + 90.0) % 180.0 - 90.0
    quantiles = np.quantile(z * 1000.0, [0.1, 0.5, 0.9]).tolist()
    return (
        center,
        valid,
        [float(value) for value in quantiles],
        float(preferred_yaw_degrees),
    )


def project_world(
    point: np.ndarray,
    camera_pose: list[float],
    intrinsics: dict[str, float],
) -> tuple[float, float] | None:
    rotation = rotation_matrix(*camera_pose[3:])
    camera = rotation.T @ (point - np.asarray(camera_pose[:3]))
    if camera[2] <= 0:
        return None
    return (
        float(camera[0] * intrinsics["fx"] / camera[2] + intrinsics["ppx"]),
        float(camera[1] * intrinsics["fy"] / camera[2] + intrinsics["ppy"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument(
        "--robotdata",
        type=Path,
        default=Path("RobotDataCollection"),
    )
    parser.add_argument(
        "--whole-object",
        action="store_true",
        help="use the detector/SAM target mask directly instead of pistol-barrel cropping",
    )
    parser.add_argument(
        "--fast-magazine",
        action="store_true",
        help="skip stochastic GraspNet and score ten PCA-relative prism grasps",
    )
    parser.add_argument(
        "--minimum-table-clearance-m",
        type=float,
        default=MINIMUM_TABLE_CLEARANCE_M,
        help="minimum virtual fingertip clearance used by this evaluation",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    minimum_table_clearance_m = float(arguments.minimum_table_clearance_m)
    if not 0.0 < minimum_table_clearance_m <= MINIMUM_TABLE_CLEARANCE_M:
        raise RuntimeError("invalid minimum table clearance")
    directory = arguments.capture_directory
    capture = json.loads((directory / "plan.json").read_text("utf-8"))
    orange_box_mode = "orange" in str(capture.get("text", "")).casefold()
    image_bgr = cv2.imread(str(directory / "left_rgb.jpg"), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("missing left RGB")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(directory / "left_depth.npy", allow_pickle=False)
    whole = np.load(directory / "target_mask.npy", allow_pickle=False).astype(bool)
    height, width = whole.shape
    whole_box = mask_bbox(whole)
    x_min, y_min, x_max, y_max = whole_box
    box_height = y_max - y_min

    gateway = LegacyApiGateway("127.0.0.1", timeout_s=180.0)
    if arguments.whole_object:
        # A magazine is already a compact, nearly convex grasp target.  The
        # detector-generated SAM mask is the desired grasp region, so a second
        # pistol-specific crop would only remove useful geometry.
        prompt_boxes = [whole_box]
        selected_index = 0
        barrel = whole.copy()
        barrel_area = int(np.count_nonzero(barrel))
        outside_area = 0
    else:
        # Locate the long lower stem of the L-shaped pistol mask.  The bottom
        # part contains the barrel/slide but not the handle.  SAM2 then supplies
        # the actual object boundary; the corridor prevents it expanding back
        # into the handle and trigger guard.
        bottom_start = int(y_min + 0.68 * box_height)
        _, bottom_columns = np.nonzero(
            whole & (np.indices(whole.shape)[0] >= bottom_start)
        )
        if len(bottom_columns) < 500:
            raise RuntimeError("cannot identify the pistol barrel stem")
        stem_left = max(0, int(np.quantile(bottom_columns, 0.02)) - 12)
        stem_right = min(width, int(np.quantile(bottom_columns, 0.98)) + 13)
        stem_top = max(y_min, int(y_min + 0.28 * box_height))
        prompt_boxes = [
            [stem_left, stem_top, stem_right, height - 8],
            [max(0, stem_left + 5), stem_top + 15, min(width, stem_right - 5), height - 20],
            [max(0, stem_left - 8), stem_top + 35, min(width, stem_right + 8), height - 35],
        ]
        segmented = gateway.segment(
            image_bytes=image.tobytes(),
            image_shape=image.shape,
            boxes_xyxy=prompt_boxes,
        )
        candidates = [
            np.asarray(value, dtype=bool) for value in segmented["masks"]
        ]
        if not candidates:
            raise RuntimeError("SAM2 returned no barrel masks")
        roi = np.zeros_like(whole)
        roi[
            stem_top : height - 18,
            max(0, stem_left - 5) : min(width, stem_right + 5),
        ] = True
        scored = []
        for index, candidate in enumerate(candidates):
            core = candidate & whole & roi
            area = int(np.count_nonzero(core))
            outside = int(np.count_nonzero(candidate & ~whole))
            scored.append((area - 2 * outside, index, core, area, outside))
        _, selected_index, barrel, barrel_area, outside_area = max(
            scored, key=lambda item: item[0]
        )
        if barrel_area < 1000:
            raise RuntimeError("SAM2 barrel core is too small")
    barrel = cv2.morphologyEx(
        barrel.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    ).astype(bool)

    intrinsics = {
        key: float(value)
        for key, value in {
            "width": width,
            "height": height,
            "ppx": 308.3497314453125,
            "ppy": 239.98605346679688,
            "fx": 607.5458374023438,
            "fy": 607.83642578125,
        }.items()
    }
    # The current device's factory values are stable and were recorded in the
    # preceding captures.  Keep them explicit in the artifact for audit.
    camera_pose = [float(value) for value in capture["current_camera_pose"]]
    center, valid_core, depth_quantiles, preferred_yaw_degrees = world_center(
        barrel, depth, intrinsics, camera_pose
    )
    context = cv2.dilate(valid_core.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    masked_depth = np.where(context, depth, 0).astype(np.uint16)
    np.save(directory / "barrel_sam_mask.npy", barrel, allow_pickle=False)
    np.save(directory / "barrel_masked_depth.npy", masked_depth, allow_pickle=False)

    if arguments.fast_magazine:
        if not arguments.whole_object:
            raise RuntimeError("--fast-magazine requires --whole-object")
        response = {"pose": [], "gdepth": [], "gwidth": [], "index": []}
    else:
        response = gateway.propose_grasps(
            depth_images=[masked_depth.tolist()],
            camera_poses=[camera_pose],
            intrinsics=[intrinsics],
            center_world_m=center.tolist(),
            debug=False,
        )
    raw_path = directory / "barrel_graspnet_response.json"
    raw_path.write_text(json.dumps(response, indent=2), "utf-8")

    distance_to_mask = cv2.distanceTransform((~barrel).astype(np.uint8), cv2.DIST_L2, 5)
    validated = []
    overlay = image_bgr.copy()
    for rank, (pose, grasp_depth, grasp_width) in enumerate(
        zip(response["pose"], response["gdepth"], response["gwidth"])
    ):
        position = np.asarray(pose[:3], dtype=float)
        grasp_rotation = rotation_matrix(*pose[3:])
        center_pixel = project_world(position, camera_pose, intrinsics)
        finger_a = position + grasp_rotation[:, 1] * float(grasp_width) / 2.0
        finger_b = position - grasp_rotation[:, 1] * float(grasp_width) / 2.0
        pixel_a = project_world(finger_a, camera_pose, intrinsics)
        pixel_b = project_world(finger_b, camera_pose, intrinsics)
        center_distance_px = float("inf")
        line_fraction = 0.0
        if center_pixel is not None:
            u, v = (int(round(center_pixel[0])), int(round(center_pixel[1])))
            if 0 <= u < width and 0 <= v < height:
                center_distance_px = float(distance_to_mask[v, u])
        if pixel_a is not None and pixel_b is not None:
            samples_u = np.linspace(pixel_a[0], pixel_b[0], 121).round().astype(int)
            samples_v = np.linspace(pixel_a[1], pixel_b[1], 121).round().astype(int)
            inside = (
                (samples_u >= 0)
                & (samples_u < width)
                & (samples_v >= 0)
                & (samples_v < height)
            )
            if np.any(inside):
                line_fraction = float(np.mean(barrel[samples_v[inside], samples_u[inside]]))
        # GraspNet's width is a recommended pre-close opening.  The legacy
        # hardware path does not command that value; it opens the gripper fully.
        # Estimate the actual object chord from the SAM-mask portion of the
        # candidate's closing line, and require clearance at the real 65.4 mm
        # opening.  Keep the network-width check as a separate diagnostic.
        estimated_object_width_m = float(grasp_width) * line_fraction
        network_width_within_limit = float(grasp_width) <= MAX_GRIPPER_WIDTH_M
        physical_fit = 0.015 <= estimated_object_width_m <= 0.058
        opening_margin_m = MAX_GRIPPER_WIDTH_M - estimated_object_width_m
        topdown_ok = float(grasp_rotation[2, 0]) <= -0.55
        near_barrel = center_distance_px <= 12.0
        crosses_barrel = line_fraction >= 0.20
        center_world_distance = float(np.linalg.norm(position - center))
        left_arm_base_distance = float(
            np.linalg.norm(position - LEFT_ARM_BASE_WORLD_M)
        )
        plausible = (
            physical_fit
            and topdown_ok
            and near_barrel
            and crosses_barrel
            and center_world_distance <= 0.08
        )
        item = {
            "rank": rank,
            "api_index": int(response["index"][rank]),
            "pose": [float(value) for value in pose],
            "gdepth_m": float(grasp_depth),
            "gwidth_m": float(grasp_width),
            "network_width_within_limit": network_width_within_limit,
            "estimated_object_chord_m": estimated_object_width_m,
            "physical_opening_margin_m": opening_margin_m,
            "approach_world_z": float(grasp_rotation[2, 0]),
            "center_pixel": list(center_pixel) if center_pixel is not None else None,
            "center_distance_to_barrel_px": center_distance_px,
            "finger_line_mask_fraction": line_fraction,
            "center_world_distance_m": center_world_distance,
            "left_arm_base_distance_m": left_arm_base_distance,
            "physical_fit": physical_fit,
            "topdown_ok": topdown_ok,
            "near_barrel": near_barrel,
            "crosses_barrel": crosses_barrel,
            "plausible": plausible,
        }
        validated.append(item)
        if center_pixel is not None:
            point = tuple(int(round(value)) for value in center_pixel)
            color = (0, 220, 0) if plausible else (0, 0, 255)
            cv2.circle(overlay, point, 5, color, -1)
            cv2.putText(overlay, str(rank), point, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if arguments.whole_object:
        # A magazine is a regular, flat rectangular prism.  GraspNet often
        # returns heavily tilted poses for it, so add deterministic vertical
        # grasps through the measured object centre.  Sweeping only the yaw
        # lets the mask-width check retain orientations that close across the
        # short side, while the simulator chooses a reachable wrist posture.
        yaw_candidates = (
            tuple(
                preferred_yaw_degrees + offset
                for offset in (0.0, -18.0, 18.0, -36.0, 36.0, -54.0, 54.0, -72.0, 72.0, 90.0)
            )
            if arguments.fast_magazine
            else (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90)
        )
        for raw_yaw_degrees in yaw_candidates:
            yaw_degrees = (float(raw_yaw_degrees) + 90.0) % 180.0 - 90.0
            rank = len(validated)
            yaw = math.radians(yaw_degrees)
            pose = [
                float(center[0]),
                float(center[1]),
                float(center[2]),
                0.0,
                math.pi / 2.0,
                yaw,
            ]
            grasp_rotation = rotation_matrix(*pose[3:])
            position = np.asarray(pose[:3], dtype=float)
            center_pixel = project_world(position, camera_pose, intrinsics)
            finger_a = position + grasp_rotation[:, 1] * MAX_GRIPPER_WIDTH_M / 2.0
            finger_b = position - grasp_rotation[:, 1] * MAX_GRIPPER_WIDTH_M / 2.0
            pixel_a = project_world(finger_a, camera_pose, intrinsics)
            pixel_b = project_world(finger_b, camera_pose, intrinsics)
            line_fraction = 0.0
            if pixel_a is not None and pixel_b is not None:
                samples_u = np.linspace(pixel_a[0], pixel_b[0], 121).round().astype(int)
                samples_v = np.linspace(pixel_a[1], pixel_b[1], 121).round().astype(int)
                inside = (
                    (samples_u >= 0)
                    & (samples_u < width)
                    & (samples_v >= 0)
                    & (samples_v < height)
                )
                if np.any(inside):
                    line_fraction = float(
                        np.mean(barrel[samples_v[inside], samples_u[inside]])
                    )
            estimated_object_width_m = MAX_GRIPPER_WIDTH_M * line_fraction
            physical_fit = 0.008 <= estimated_object_width_m <= 0.063
            crosses_barrel = line_fraction >= 0.12
            yaw_offset = abs(
                (yaw_degrees - preferred_yaw_degrees + 90.0) % 180.0 - 90.0
            )
            short_side_direction_ok = not orange_box_mode or yaw_offset <= 20.0
            plausible = physical_fit and crosses_barrel and short_side_direction_ok
            item = {
                "rank": rank,
                "api_index": -1 - rank,
                "candidate_source": "analytic_topdown_prism",
                "yaw_degrees": yaw_degrees,
                "pose": pose,
                "gdepth_m": 0.0,
                "gwidth_m": MAX_GRIPPER_WIDTH_M,
                "network_width_within_limit": True,
                "estimated_object_chord_m": estimated_object_width_m,
                "physical_opening_margin_m": (
                    MAX_GRIPPER_WIDTH_M - estimated_object_width_m
                ),
                "approach_world_z": -1.0,
                "center_pixel": (
                    list(center_pixel) if center_pixel is not None else None
                ),
                "center_distance_to_barrel_px": 0.0,
                "finger_line_mask_fraction": line_fraction,
                "center_world_distance_m": 0.0,
                "left_arm_base_distance_m": float(
                    np.linalg.norm(position - LEFT_ARM_BASE_WORLD_M)
                ),
                "physical_fit": physical_fit,
                "topdown_ok": True,
                "near_barrel": True,
                "crosses_barrel": crosses_barrel,
                "short_side_yaw_error_deg": yaw_offset,
                "short_side_direction_ok": short_side_direction_ok,
                "plausible": plausible,
            }
            if arguments.fast_magazine:
                item["fast_candidate_score"] = 1.0 - yaw_offset / 90.0
            validated.append(item)
            if center_pixel is not None:
                point = tuple(int(round(value)) for value in center_pixel)
                color = (255, 0, 255) if plausible else (120, 0, 120)
                cv2.circle(overlay, point, 4, color, 1)

    mask_overlay = image_bgr.copy()
    mask_overlay[barrel] = (
        0.45 * mask_overlay[barrel] + 0.55 * np.array([255, 150, 0])
    ).astype(np.uint8)
    cv2.rectangle(mask_overlay, prompt_boxes[selected_index][:2], prompt_boxes[selected_index][2:], (255, 0, 0), 2)
    cv2.imwrite(str(directory / "barrel_sam_overlay.jpg"), mask_overlay)
    cv2.imwrite(str(directory / "barrel_grasp_candidates.jpg"), overlay)

    plausible_items = sorted(
        (item for item in validated if item["plausible"]),
        key=(
            (lambda item: (-float(item.get("fast_candidate_score", 0.0)), item["rank"]))
            if arguments.fast_magazine
            else (lambda item: (item["left_arm_base_distance_m"], item["rank"]))
        ),
    )
    simulation_results = []
    if plausible_items:
        sys.path.insert(0, str(arguments.robotdata))
        from utils.kinematic_transforms import GripperKinematics  # type: ignore

        config = json.loads(
            (arguments.robotdata / "config" / "env_config.json").read_text("utf-8")
        )["left_arm_config"]
        kinematics = GripperKinematics(
            config["base_extrinsic"]["R"],
            config["base_extrinsic"]["t"],
            config["gripper_extrinsic"]["R"],
            config["gripper_extrinsic"]["t"],
        )
        # The candidates are ordered near-to-far so the eventual selector can
        # still honor the near-side preference, but do not discard the farther
        # half before IK.  A nearby pose may be geometrically plausible yet
        # unsolvable with the required 50 mm pregrasp, while a slightly farther
        # and better-centered candidate can have a clean two-stage path.
        for item in plausible_items:
            grasp_pose = item["pose"]
            # GraspNet's pose is a geometric reference only.  The first motion
            # target is 50 mm behind it along the gripper approach axis; the
            # second simulation then advances directly to the selected final
            # insertion pose without stopping at the GraspNet pose.
            pregrasp_pose = kinematics.cal_pre_pose(
                grasp_pose, PREGRASP_RETRACTION_M
            )
            pre_end = kinematics.gripper_to_end(pregrasp_pose)
            pregrasp_clearance_m = virtual_tip_clearance_m(pregrasp_pose)
            if pregrasp_clearance_m < minimum_table_clearance_m:
                simulation_results.append(
                    {
                        "rank": item["rank"],
                        "stage1_success": False,
                        "pregrasp_retraction_m": PREGRASP_RETRACTION_M,
                        "pregrasp_gripper_pose": [
                            float(value) for value in pregrasp_pose
                        ],
                        "pregrasp_virtual_tip_clearance_m": pregrasp_clearance_m,
                        "rejection": "pregrasp virtual tip does not meet table clearance",
                    }
                )
                continue
            stage1 = gateway.simulate(
                target_pose=pre_end,
                arm="left",
                left_joints_deg=capture["left_joints"],
                right_joints_deg=capture["right_joints"],
                num_waypoints=6 if arguments.fast_magazine else 10,
                recording=not arguments.fast_magazine,
            )
            simulation = {
                "rank": item["rank"],
                "stage1_success": bool(stage1["success"]),
                "pregrasp_retraction_m": PREGRASP_RETRACTION_M,
                "pregrasp_gripper_pose": [float(value) for value in pregrasp_pose],
                "pregrasp_virtual_tip_clearance_m": pregrasp_clearance_m,
            }
            if stage1["success"]:
                simulation["stage1_waypoint_count"] = len(stage1["path"])
                simulation["approach_trials"] = []
                advances_m = candidate_advances_m(item["gdepth_m"])
                if orange_box_mode:
                    requested_depth_m = max(0.0, float(item["gdepth_m"]))
                    orange_deep_advance_m = requested_depth_m + 0.010
                    advances_m = (
                        orange_deep_advance_m,
                        *(value for value in advances_m if abs(value - orange_deep_advance_m) > 1e-9),
                    )
                for advance_m in advances_m:
                    final_gripper = kinematics.cal_pre_pose(grasp_pose, -advance_m)
                    final_end = kinematics.gripper_to_end(final_gripper)
                    virtual_clearance_m = virtual_tip_clearance_m(final_gripper)
                    if virtual_clearance_m < minimum_table_clearance_m:
                        simulation["approach_trials"].append(
                            {
                                "advance_m": advance_m,
                                "success": False,
                                "simulation_called": False,
                                "virtual_tip_clearance_m": virtual_clearance_m,
                                "minimum_table_clearance_m": minimum_table_clearance_m,
                                "rejection": "virtual tip does not meet table clearance",
                                "target_end_pose": [float(value) for value in final_end],
                            }
                        )
                        continue
                    stage2 = gateway.simulate(
                        target_pose=final_end,
                        arm="left",
                        left_joints_deg=stage1["path"][-1],
                        right_joints_deg=capture["right_joints"],
                        num_waypoints=max(
                            4,
                            int(
                                math.ceil(
                                    (PREGRASP_RETRACTION_M + advance_m) / 0.01
                                )
                            ),
                        ),
                        recording=not arguments.fast_magazine,
                    )
                    trial = {
                        "advance_m": advance_m,
                        "approach_distance_m": PREGRASP_RETRACTION_M + advance_m,
                        "success": bool(stage2["success"]),
                        "simulation_called": True,
                        "virtual_tip_clearance_m": virtual_clearance_m,
                        "minimum_table_clearance_m": minimum_table_clearance_m,
                        "final_gripper_pose": [
                            float(value) for value in final_gripper
                        ],
                        "target_end_pose": [float(value) for value in final_end],
                    }
                    simulation["approach_trials"].append(trial)
                    if not stage2["success"]:
                        continue
                    simulation["stage2_success"] = True
                    simulation["selected_advance_m"] = advance_m
                    simulation["approach_distance_m"] = (
                        PREGRASP_RETRACTION_M + advance_m
                    )
                    if not arguments.fast_magazine:
                        for stage_name, stage in (("pregrasp", stage1), ("approach", stage2)):
                            video = directory / f"candidate_{item['rank']}_{stage_name}.mp4"
                            video.write_bytes(base64.b64decode(stage["video"]))
                            simulation[f"{stage_name}_video"] = str(video)
                    simulation["waypoints"] = stage1["path"] + stage2["path"]
                    break
                else:
                    simulation["stage2_success"] = False
            simulation_results.append(simulation)

    summary: dict[str, Any] = {
        "ok": True,
        "real_robot_command_sent": False,
        "gripper_command_sent": False,
        "default_extrinsic_used": True,
        "grasp_region": "whole_object" if arguments.whole_object else "pistol_barrel",
        "gripper_clearance_policy": {
            "simulated_open_mesh_length_m": SIMULATED_OPEN_MESH_LENGTH_M,
            "kinematic_tcp_length_m": KINEMATIC_TCP_LENGTH_M,
            "virtual_collision_length_m": VIRTUAL_COLLISION_LENGTH_M,
            "minimum_table_clearance_m": minimum_table_clearance_m,
            "pregrasp_retraction_m": PREGRASP_RETRACTION_M,
            "final_insertion_bias_m": FINAL_INSERTION_BIAS_M,
        },
        "whole_pistol_bbox": whole_box,
        "sam_prompt_boxes": prompt_boxes,
        "selected_sam_prompt": selected_index,
        "barrel_mask_bbox": mask_bbox(barrel),
        "barrel_mask_pixels": int(np.count_nonzero(barrel)),
        "barrel_valid_depth_pixels": int(np.count_nonzero(valid_core)),
        "barrel_depth_quantiles_mm": depth_quantiles,
        "barrel_center_world_m": center.tolist(),
        "preferred_magazine_yaw_degrees": preferred_yaw_degrees,
        "fast_magazine": arguments.fast_magazine,
        "total_grasp_candidates": len(validated),
        "plausible_count": len(plausible_items),
        "validated_candidates": validated,
        "simulations": simulation_results,
        "reliable_simulated_candidates": [
            item["rank"]
            for item in simulation_results
            if item.get("stage1_success") and item.get("stage2_success")
        ],
    }
    (directory / "barrel_grasp_evaluation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), "utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
