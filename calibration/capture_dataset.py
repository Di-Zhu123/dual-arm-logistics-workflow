"""Manually gated, read-only robot/camera calibration data capture.

This program deliberately contains no call to ``action``, ``movej``, ``movel``,
or a gripper command. The operator positions the arm with the teach pendant,
waits for it to become stationary, and explicitly presses Enter for each sample.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys
import threading
import time
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import pyrealsense2 as rs

from .board import BoardSpec, detect_corners
from .geometry import (
    legacy_link_backward_transform,
    matrix_json,
    pose_xyz_euler_to_transform,
    rotation_error_deg,
    translation_error_m,
)
from .io import SCHEMA_VERSION, read_json, sha256_file, write_json
from .preview_three_cameras import PreviewHTTPServer, PreviewState


def parse_inner_corners(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("×", "x")
    try:
        columns, rows = (int(part) for part in normalized.split("x", 1))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("use COLUMNSxROWS, for example 9x6") from error
    if columns < 3 or rows < 3:
        raise argparse.ArgumentTypeError("checkerboard needs at least 3x3 inner corners")
    return columns, rows


def legacy_world_from_base(config: dict[str, Any], arm: str) -> np.ndarray:
    section = config[f"{arm}_arm_config"]["base_extrinsic"]
    return legacy_link_backward_transform(section["R"], section["t"])


def camera_metadata(camera: Any, color: Any, depth: Any, started_ns: int, ended_ns: int) -> dict:
    intrinsics = color.profile.as_video_stream_profile().get_intrinsics()
    device = camera.profile.get_device()
    return {
        "serial_number": device.get_info(rs.camera_info.serial_number),
        "host_capture_started_ns": started_ns,
        "host_capture_ended_ns": ended_ns,
        "color_frame_number": int(color.get_frame_number()),
        "depth_frame_number": int(depth.get_frame_number()),
        "color_timestamp_ms": float(color.get_timestamp()),
        "depth_timestamp_ms": float(depth.get_timestamp()),
        "timestamp_domain": str(color.get_frame_timestamp_domain()),
        "depth_scale_m_per_unit": float(depth.get_units()),
        "intrinsics": {
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "ppx": float(intrinsics.ppx),
            "ppy": float(intrinsics.ppy),
            "distortion_model": str(intrinsics.model),
            "coeffs": [float(value) for value in intrinsics.coeffs],
        },
    }


def capture_camera(camera: Any, settle_frames: int) -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(settle_frames):
        camera.pipeline.wait_for_frames(5000)
    started_ns = time.time_ns()
    frames = camera.pipeline.wait_for_frames(5000)
    aligned = camera.align.process(frames)
    depth = aligned.get_depth_frame()
    color = aligned.get_color_frame()
    ended_ns = time.time_ns()
    if not depth or not color:
        raise RuntimeError("camera returned an incomplete aligned frameset")
    image_bgr = np.asanyarray(color.get_data()).copy()
    depth_raw = np.asanyarray(depth.get_data()).copy()
    return image_bgr, depth_raw, camera_metadata(camera, color, depth, started_ns, ended_ns)


class LiveCaptureCoordinator:
    """Continuously acquire one synchronized triplet for preview and sampling.

    This is the only code path that calls ``wait_for_frames`` while integrated
    preview is enabled.  The sampling thread requests a triplet captured after
    the Enter key, so preview and dataset capture never race for a pipeline.
    """

    def __init__(
        self,
        camera_objects: dict[str, Any],
        preview_state: PreviewState,
        settle_frames: int,
    ) -> None:
        self.camera_objects = camera_objects
        self.preview_state = preview_state
        self.settle_frames = settle_frames
        self.condition = threading.Condition()
        self.sequence = 0
        self.latest: dict[str, tuple[np.ndarray, np.ndarray, dict]] | None = None
        self.last_error: Exception | None = None
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self) -> None:
        first = True
        with ThreadPoolExecutor(max_workers=3) as executor:
            while not self.stop.is_set():
                try:
                    settle = self.settle_frames if first else 0
                    pending = {
                        name: executor.submit(capture_camera, camera, settle)
                        for name, camera in self.camera_objects.items()
                    }
                    captures = {
                        name: future.result() for name, future in pending.items()
                    }
                    first = False
                    images = {name: value[0] for name, value in captures.items()}
                    frame_numbers = {
                        name: int(value[2]["color_frame_number"])
                        for name, value in captures.items()
                    }
                    self.preview_state.update(images, frame_numbers)
                    with self.condition:
                        self.latest = captures
                        self.last_error = None
                        self.sequence += 1
                        self.condition.notify_all()
                except Exception as error:
                    self.preview_state.error(error)
                    with self.condition:
                        self.last_error = error
                        self.condition.notify_all()
                    time.sleep(0.2)

    def snapshot_after(
        self, minimum_started_ns: int, timeout_s: float = 10.0
    ) -> dict[str, tuple[np.ndarray, np.ndarray, dict]]:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while True:
                if self.latest is not None and all(
                    value[2]["host_capture_started_ns"] >= minimum_started_ns
                    for value in self.latest.values()
                ):
                    return self.latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f": {self.last_error}" if self.last_error else ""
                    raise RuntimeError(f"timed out waiting for live camera triplet{detail}")
                self.condition.wait(timeout=remaining)

    def close(self) -> None:
        self.stop.set()
        self.preview_state.stop.set()
        self.worker.join(timeout=6.0)


def read_joints(arm: Any, name: str) -> tuple[float, ...]:
    status, joints = arm.get_joint_degree()
    if status != 0:
        raise RuntimeError(f"failed to read {name} joints, status={status}")
    values = tuple(float(value) for value in joints)
    if len(values) != 7:
        raise RuntimeError(f"{name} arm returned {len(values)} joints")
    return values


def next_missing_index(completed_indices: set[int], sample_count: int) -> int | None:
    return next(
        (index for index in range(sample_count) if index not in completed_indices),
        None,
    )


def write_sample(
    sample_directory: Path,
    sample_id: str,
    captures: dict[str, tuple[np.ndarray, np.ndarray, dict]],
    robot: dict[str, Any],
) -> None:
    # A unique sibling makes an interrupted write recoverable and avoids deleting
    # any previous operator data. Only the final atomic rename publishes a sample.
    temporary = sample_directory.with_name(
        f".{sample_directory.name}.{uuid4().hex}.partial"
    )
    temporary.mkdir(parents=True)
    cameras: dict[str, Any] = {}
    for name, (image_bgr, depth_raw, metadata) in captures.items():
        rgb_name = f"{name}_rgb.png"
        depth_name = f"{name}_depth.npy"
        if not cv2.imwrite(str(temporary / rgb_name), image_bgr):
            raise OSError(f"failed to write {rgb_name}")
        np.save(temporary / depth_name, depth_raw, allow_pickle=False)
        cameras[name] = {
            **metadata,
            "rgb_file": rgb_name,
            "depth_file": depth_name,
            "rgb_sha256": sha256_file(temporary / rgb_name),
            "depth_sha256": sha256_file(temporary / depth_name),
        }
    write_json(
        temporary / "sample.json",
        {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "cameras": cameras,
            "robot": robot,
        },
    )
    if sample_directory.exists():
        raise FileExistsError(sample_directory)
    temporary.rename(sample_directory)


def close_environment(environment: Any) -> None:
    try:
        if hasattr(environment, "head_camera"):
            environment.head_camera.close()
    finally:
        environment.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotdata-root", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument(
        "--camera-identities",
        required=True,
        help="JSON mapping the physical left/head/right roles to RealSense serials",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "head_eye_to_hand",
            "head_from_wrist_handheld",
            "wrist_eye_in_hand_via_head",
            "wrist_eye_in_hand_fixed_board",
            "alignment",
        ),
        required=True,
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--inner-corners", type=parse_inner_corners, required=True)
    parser.add_argument("--square-size-mm", type=float, required=True)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--settle-frames", type=int, default=3)
    parser.add_argument("--max-joint-drift-deg", type=float, default=0.05)
    parser.add_argument("--max-camera-skew-ms", type=float, default=80.0)
    parser.add_argument(
        "--preview-http-host",
        default="127.0.0.1",
        help="integrated preview bind address; keep localhost and use an SSH tunnel",
    )
    parser.add_argument(
        "--preview-http-port",
        type=int,
        default=0,
        help="enable integrated three-camera web preview on this port; 0 disables it",
    )
    parser.add_argument("--ack-read-only-capture", action="store_true")
    arguments = parser.parse_args()
    if not arguments.ack_read_only_capture:
        parser.error("--ack-read-only-capture is required")
    if not 8 <= arguments.samples <= 100:
        parser.error("--samples must be between 8 and 100")

    board = BoardSpec(*arguments.inner_corners, arguments.square_size_mm / 1000.0)
    root = Path(arguments.output).resolve()
    samples_root = root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    config = read_json(arguments.env_config)
    identity_document = read_json(arguments.camera_identities)
    expected_identities = identity_document.get("cameras", {})
    try:
        expected_serials = {
            role: str(expected_identities[role]["serial_number"])
            for role in ("left", "head", "right")
        }
    except (KeyError, TypeError) as error:
        parser.error(
            "--camera-identities must map left/head/right to serial_number values"
        )
    if len(set(expected_serials.values())) != 3:
        parser.error("--camera-identities must contain three distinct serial numbers")
    if arguments.settle_frames < 0:
        parser.error("--settle-frames cannot be negative")
    if arguments.max_joint_drift_deg <= 0:
        parser.error("--max-joint-drift-deg must be positive")
    if arguments.max_camera_skew_ms <= 0:
        parser.error("--max-camera-skew-ms must be positive")
    if arguments.preview_http_port != 0 and not 1024 <= arguments.preview_http_port <= 65535:
        parser.error("--preview-http-port must be 0 or between 1024 and 65535")
    robotdata_root = Path(arguments.robotdata_root).resolve()
    sys.path.insert(0, str(robotdata_root))
    from utils.arm_environment import DualArmEnvironment  # type: ignore

    try:
        environment = DualArmEnvironment(
            config["left_arm_config"],
            config["right_arm_config"],
            config["head_camera_config"],
        )
    except RuntimeError as error:
        detail = str(error)
        if "xioctl" in detail.lower() or "device or resource busy" in detail.lower():
            raise RuntimeError(
                "RealSense device is busy; stop the read-only preview process "
                "and retry"
            ) from error
        raise
    selected_arm = environment.arm_left if arguments.arm == "left" else environment.arm_right
    camera_objects = {
        "left": environment.arm_left.camera,
        "head": environment.head_camera,
        "right": environment.arm_right.camera,
    }
    serials = {
        name: camera.profile.get_device().get_info(rs.camera_info.serial_number)
        for name, camera in camera_objects.items()
    }
    if serials != expected_serials:
        close_environment(environment)
        raise RuntimeError(
            "camera role/serial mismatch; refusing to collect mislabeled data: "
            f"expected={expected_serials}, actual={serials}. "
            "Do not edit calibration data to work around this; inspect cabling/enumeration."
        )
    expected_device_ids = {
        "left": int(config["left_arm_config"]["device_id"]),
        "head": int(config["head_camera_config"]["device_id"]),
        "right": int(config["right_arm_config"]["device_id"]),
    }
    manifest_path = root / "dataset.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": arguments.mode,
        "arm": arguments.arm,
        "board": board.canonical(),
        "camera_serials": serials,
        "enumeration_device_ids_at_capture": expected_device_ids,
        "world_from_arm_base": matrix_json(legacy_world_from_base(config, arguments.arm)),
        "coordinate_convention": "p_parent = parent_from_child @ p_child; column vectors",
        "safety": {
            "robot_commands_issued_by_this_program": False,
            "operator_positions_arm_with_teach_pendant": True,
            "maximum_joint_drift_deg": arguments.max_joint_drift_deg,
            "maximum_camera_host_midpoint_skew_ms": arguments.max_camera_skew_ms,
        },
    }
    if manifest_path.exists():
        existing = read_json(manifest_path)
        for field in ("mode", "arm", "board", "camera_serials"):
            if existing.get(field) != manifest[field]:
                close_environment(environment)
                raise RuntimeError(f"existing dataset has a different {field}")
    else:
        write_json(manifest_path, manifest)

    preview_state: PreviewState | None = None
    preview_coordinator: LiveCaptureCoordinator | None = None
    preview_server: PreviewHTTPServer | None = None
    preview_server_thread: threading.Thread | None = None
    if arguments.preview_http_port:
        preview_state = PreviewState()
        preview_coordinator = LiveCaptureCoordinator(
            camera_objects, preview_state, arguments.settle_frames
        )
        try:
            preview_server = PreviewHTTPServer(
                (arguments.preview_http_host, arguments.preview_http_port), preview_state
            )
        except Exception:
            preview_coordinator.close()
            close_environment(environment)
            raise
        preview_server_thread = threading.Thread(
            target=preview_server.serve_forever,
            kwargs={"poll_interval": 0.2},
            daemon=True,
        )
        preview_server_thread.start()
        print(
            "Integrated preview URL: "
            f"http://{arguments.preview_http_host}:{arguments.preview_http_port}/"
        )

    print("READ-ONLY CAPTURE MODE: this program never commands either arm or gripper.")
    if arguments.mode == "head_eye_to_hand":
        print(f"Checkerboard must be rigidly attached to the {arguments.arm} tool.")
    elif arguments.mode == "head_from_wrist_handheld":
        print(
            "Operator may move the checkerboard freely between samples, but must hold it "
            "stationary and fully visible in left/head/right when pressing Enter."
        )
        print("Both robot arms must remain stationary; this program does not move them.")
    elif arguments.mode == "wrist_eye_in_hand_via_head":
        print(
            "The checkerboard may move freely between samples, but it must be held "
            "stationary and fully visible in left/head/right when pressing Enter."
        )
        print(
            "Between samples, use the teach pendant to give BOTH arm ends diverse "
            "positions and orientations. This program never commands either arm."
        )
    elif arguments.mode == "wrist_eye_in_hand_fixed_board":
        print(
            "The checkerboard must be rigidly fixed and remain motionless for the "
            "entire run."
        )
        print(
            "Use the teach pendant to give the selected arm end diverse positions "
            "and orientations between samples. This program never commands the arm."
        )
    else:
        print("Checkerboard must remain fixed in the workcell and visible in all three cameras.")
    try:
        completed_indices = {
            int(path.name.removeprefix("sample_"))
            for path in samples_root.glob("sample_[0-9][0-9][0-9]")
            if path.is_dir()
            and (path / "sample.json").is_file()
            and path.name.removeprefix("sample_").isdigit()
        }
        print(
            f"Resume status: {len(completed_indices)}/{arguments.samples} samples already saved."
        )
        last_saved_end_transforms: dict[str, np.ndarray] = {}
        while True:
            index = next_missing_index(completed_indices, arguments.samples)
            if index is None:
                print(f"Capture complete: {arguments.samples}/{arguments.samples} samples saved.")
                break
            if preview_state is not None:
                preview_state.set_operator_status(
                    f"saved {len(completed_indices)}/{arguments.samples}; "
                    f"prepare sample_{index:03d}, then press Enter in terminal"
                )
            if arguments.mode == "head_from_wrist_handheld":
                prompt = (
                    f"Hold board pose {index + 1}/{arguments.samples} still and visible in all "
                    "three cameras, then press Enter (or type q to quit): "
                )
            elif arguments.mode == "wrist_eye_in_hand_via_head":
                prompt = (
                    f"Set diverse stationary arm pose {index + 1}/{arguments.samples}; "
                    "hold board still in all three views, then press Enter "
                    "(or type q to quit): "
                )
            elif arguments.mode == "wrist_eye_in_hand_fixed_board":
                prompt = (
                    f"Set the selected arm's diverse stationary pose {index + 1}/{arguments.samples}; "
                    "keep the fixed board visible in all three views, then press Enter "
                    "(or type q to quit): "
                )
            else:
                prompt = (
                    f"Position pose {index + 1}/{arguments.samples}, stop the arm, then press "
                    "Enter (or type q to quit): "
                )
            response = input(prompt).strip().lower()
            if response == "q":
                break
            before_left = read_joints(environment.arm_left, "left")
            before_right = read_joints(environment.arm_right, "right")
            current_end_transforms = {
                "left": pose_xyz_euler_to_transform(
                    environment.arm_left.gripper_kinematics.get_arm_end_forward(
                        before_left
                    )
                ),
                "right": pose_xyz_euler_to_transform(
                    environment.arm_right.gripper_kinematics.get_arm_end_forward(
                        before_right
                    )
                ),
            }
            if arguments.mode in (
                "wrist_eye_in_hand_via_head",
                "wrist_eye_in_hand_fixed_board",
            ) and last_saved_end_transforms:
                pose_delta = {
                    arm: {
                        "translation_mm": translation_error_m(
                            current_end_transforms[arm], last_saved_end_transforms[arm]
                        )
                        * 1000.0,
                        "rotation_deg": rotation_error_deg(
                            current_end_transforms[arm], last_saved_end_transforms[arm]
                        ),
                    }
                    for arm in ("left", "right")
                }
                if max(value["rotation_deg"] for value in pose_delta.values()) < 1.0:
                    warning = (
                        f"WARNING sample_{index:03d}: both arm ends changed less than "
                        "1 degree since the previous saved sample; change the arm pose "
                        "before pressing Enter."
                    )
                    print(warning)
                    if preview_state is not None:
                        preview_state.set_operator_status(warning)
                else:
                    status = (
                        f"sample_{index:03d} pose delta: "
                        f"left {pose_delta['left']['translation_mm']:.1f}mm/"
                        f"{pose_delta['left']['rotation_deg']:.1f}deg, "
                        f"right {pose_delta['right']['translation_mm']:.1f}mm/"
                        f"{pose_delta['right']['rotation_deg']:.1f}deg"
                    )
                    print(status)
                    if preview_state is not None:
                        preview_state.set_operator_status(status)
            capture_requested_ns = time.time_ns()
            try:
                if preview_coordinator is not None:
                    captures = preview_coordinator.snapshot_after(capture_requested_ns)
                else:
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        pending = {
                            name: executor.submit(
                                capture_camera, camera, arguments.settle_frames
                            )
                            for name, camera in camera_objects.items()
                        }
                        captures = {
                            name: future.result() for name, future in pending.items()
                        }
            except RuntimeError as error:
                message = f"Rejected sample_{index:03d}: camera capture failed: {error}; retry."
                print(message)
                if preview_state is not None:
                    preview_state.set_operator_status(message)
                continue
            after_left = read_joints(environment.arm_left, "left")
            after_right = read_joints(environment.arm_right, "right")
            drift = max(
                abs(before - after)
                for before, after in zip(before_left + before_right, after_left + after_right)
            )
            if drift > arguments.max_joint_drift_deg:
                message = (
                    f"Rejected sample_{index:03d}: joint drift {drift:.4f}deg exceeds "
                    "limit; retry the same sample number."
                )
                print(message)
                if preview_state is not None:
                    preview_state.set_operator_status(message)
                continue
            capture_midpoints_ns = {
                name: (metadata["host_capture_started_ns"] + metadata["host_capture_ended_ns"])
                / 2.0
                for name, (_, _, metadata) in captures.items()
            }
            camera_skew_ms = (
                max(capture_midpoints_ns.values()) - min(capture_midpoints_ns.values())
            ) / 1_000_000.0
            if camera_skew_ms > arguments.max_camera_skew_ms:
                message = (
                    f"Rejected sample_{index:03d}: camera host midpoint skew "
                    f"{camera_skew_ms:.2f}ms exceeds "
                    f"{arguments.max_camera_skew_ms:.2f}ms; retry the same sample number."
                )
                print(message)
                if preview_state is not None:
                    preview_state.set_operator_status(message)
                continue
            try:
                corners = detect_corners(captures["head"][0], board)
            except ValueError as error:
                message = (
                    f"Rejected sample_{index:03d}: head checkerboard detection failed: "
                    f"{error}; retry the same sample number."
                )
                print(message)
                if preview_state is not None:
                    preview_state.set_operator_status(message)
                continue
            if arguments.mode in (
                "head_from_wrist_handheld",
                "wrist_eye_in_hand_via_head",
                "alignment",
            ):
                missing = []
                for name in ("left", "right"):
                    try:
                        detect_corners(captures[name][0], board)
                    except ValueError:
                        missing.append(name)
                if missing:
                    message = (
                        f"Rejected sample_{index:03d}: checkerboard not detected in "
                        f"{', '.join(missing)}; retry the same sample number."
                    )
                    print(message)
                    if preview_state is not None:
                        preview_state.set_operator_status(message)
                    continue

            selected_joints = before_left if arguments.arm == "left" else before_right
            base_from_tool = pose_xyz_euler_to_transform(
                selected_arm.gripper_kinematics.get_arm_end_pose(selected_joints)
            )
            dynamic_world_from_cameras = {
                "left": matrix_json(
                    pose_xyz_euler_to_transform(
                        environment.arm_left.camera_kinematics.get_camera_forward(before_left)
                    )
                ),
                "right": matrix_json(
                    pose_xyz_euler_to_transform(
                        environment.arm_right.camera_kinematics.get_camera_forward(before_right)
                    )
                ),
            }
            dynamic_world_from_arm_ends = {
                arm: matrix_json(current_end_transforms[arm]) for arm in ("left", "right")
            }
            sample_id = f"sample_{index:03d}"
            robot = {
                "arm": arguments.arm,
                "left_joints_before_deg": list(before_left),
                "left_joints_after_deg": list(after_left),
                "right_joints_before_deg": list(before_right),
                "right_joints_after_deg": list(after_right),
                "maximum_joint_drift_deg": drift,
                "camera_host_midpoint_skew_ms": camera_skew_ms,
                "base_from_tool": matrix_json(base_from_tool),
                "dynamic_world_from_arm_ends": dynamic_world_from_arm_ends,
                "dynamic_world_from_cameras": dynamic_world_from_cameras,
            }
            write_sample(samples_root / sample_id, sample_id, captures, robot)
            completed_indices.add(index)
            last_saved_end_transforms = current_end_transforms
            print(f"Saved {sample_id}; head corners={len(corners)}, drift={drift:.4f}deg")
            if preview_state is not None:
                preview_state.set_operator_status(
                    f"accepted {sample_id}; saved "
                    f"{len(completed_indices)}/{arguments.samples}"
                )
    finally:
        if preview_server is not None:
            preview_server.shutdown()
            preview_server.server_close()
        if preview_server_thread is not None:
            preview_server_thread.join(timeout=3.0)
        if preview_coordinator is not None:
            preview_coordinator.close()
        close_environment(environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
