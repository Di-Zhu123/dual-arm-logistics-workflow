"""Defensive client for the existing four-byte-length-prefixed JSON APIs."""

from __future__ import annotations

import base64
import json
import socket
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import ProtocolError


DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ProtocolError(
                f"connection closed with {remaining} of {size} response bytes missing"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class LengthPrefixedJsonClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_s: float = 30.0,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        if not host or not 1 <= port <= 65535:
            raise ValueError("valid host and port are required")
        if timeout_s <= 0 or max_message_bytes <= 0:
            raise ValueError("timeout and message limit must be positive")
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.max_message_bytes = max_message_bytes

    def request(self, payload: Mapping[str, Any]) -> Any:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        if len(encoded) > self.max_message_bytes:
            raise ProtocolError("request exceeds configured message limit")
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as connection:
            connection.settimeout(self.timeout_s)
            connection.sendall(struct.pack(">I", len(encoded)) + encoded)
            response_size = struct.unpack(">I", _receive_exact(connection, 4))[0]
            if response_size > self.max_message_bytes:
                raise ProtocolError("response exceeds configured message limit")
            response = _receive_exact(connection, response_size)
        try:
            return json.loads(response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolError("response is not valid UTF-8 JSON") from error


def build_open_vocabulary_request(
    *,
    image_bytes: bytes,
    image_shape: Sequence[int],
    text: str,
) -> dict[str, Any]:
    if len(image_shape) != 3 or not text.strip():
        raise ValueError("an HWC image shape and non-empty text are required")
    return {
        "image": base64.b64encode(image_bytes).decode("ascii"),
        "image_shape": [int(value) for value in image_shape],
        # det_api.py supports seg and seg_open; the old task='ovd' is invalid.
        "task": "seg_open",
        "text": text,
    }


def build_segmentation_request(
    *,
    image_bytes: bytes,
    image_shape: Sequence[int],
    boxes_xyxy: Sequence[Sequence[float]],
) -> dict[str, Any]:
    if len(image_shape) != 3:
        raise ValueError("an HWC image shape is required")
    if any(len(box) != 4 for box in boxes_xyxy):
        raise ValueError("each segmentation box must contain xyxy")
    return {
        "image": base64.b64encode(image_bytes).decode("ascii"),
        "image_shape": [int(value) for value in image_shape],
        "task": "seg",
        "boxes": [[float(value) for value in box] for box in boxes_xyxy],
    }


def build_grasp_request(
    *,
    depth_images: Sequence[Any],
    camera_poses: Sequence[Sequence[float]],
    intrinsics: Sequence[Mapping[str, Any]],
    center_world_m: Sequence[float],
    debug: bool = False,
) -> dict[str, Any]:
    if not (len(depth_images) == len(camera_poses) == len(intrinsics)):
        raise ValueError("depth, pose, and intrinsics camera counts must match")
    if not depth_images or len(center_world_m) != 3:
        raise ValueError("at least one depth image and a three-value center are required")
    if any(len(pose) != 6 for pose in camera_poses):
        raise ValueError("each camera pose must contain xyz and Euler rx/ry/rz")
    request: dict[str, Any] = {
        "task": "grasp",
        "depth": list(depth_images),
        "pose": [[float(value) for value in pose] for pose in camera_poses],
        "intrinsics": [dict(value) for value in intrinsics],
        "center": [float(value) for value in center_world_m],
    }
    # api2.py enables its very large debug payload by key presence, not truthiness.
    # Therefore debug=False must omit this key for compatibility with that bug.
    if debug:
        request["debug"] = True
    return request


def build_simulation_request(
    *,
    target_pose: Sequence[float],
    target_joints_deg: Sequence[float] | None = None,
    arm: str,
    left_joints_deg: Sequence[float],
    right_joints_deg: Sequence[float],
    num_waypoints: int,
    recording: bool = True,
) -> dict[str, Any]:
    if arm not in {"left", "right", "0", "1"}:
        raise ValueError("arm must be left/right or the legacy 0/1 identifier")
    arm_id = 0 if arm in {"left", "0"} else 1
    if len(target_pose) != 6 or len(left_joints_deg) != 7 or len(right_joints_deg) != 7:
        raise ValueError("simulation requires a 6D pose and two 7-joint states")
    if target_joints_deg is not None and len(target_joints_deg) != 7:
        raise ValueError("simulation joint target must contain seven values")
    if num_waypoints <= 0:
        raise ValueError("num_waypoints must be positive")
    request = {
        "task": "sim",
        "pose": [float(value) for value in target_pose],
        "arm": arm_id,
        "joints_left": [float(value) for value in left_joints_deg],
        "joints_right": [float(value) for value in right_joints_deg],
        "num_waypoints": int(num_waypoints),
        "recording": bool(recording),
    }
    if target_joints_deg is not None:
        request["target_joints"] = [float(value) for value in target_joints_deg]
    return request


def build_simulation_sequence_request(
    *,
    targets: Sequence[Mapping[str, Any]],
    arm: str,
    left_joints_deg: Sequence[float],
    right_joints_deg: Sequence[float],
    recording: bool = True,
) -> dict[str, Any]:
    """Build one request that plans several targets in one persistent scene.

    The simulation service already owns one long-lived Genesis scene.  This
    request shape lets a workflow reuse that scene for a complete staged motion
    instead of opening one request (and one video) per stage.
    """
    if arm not in {"left", "right", "0", "1"}:
        raise ValueError("arm must be left/right or the legacy 0/1 identifier")
    if len(left_joints_deg) != 7 or len(right_joints_deg) != 7:
        raise ValueError("simulation requires two 7-joint states")
    if not targets:
        raise ValueError("simulation sequence requires at least one target")
    normalized_targets = []
    for index, target in enumerate(targets):
        pose = target.get("pose")
        joints = target.get("target_joints")
        num_waypoints = int(target.get("num_waypoints", 0))
        if not isinstance(pose, Sequence) or len(pose) != 6:
            raise ValueError(f"simulation sequence target {index} needs a 6D pose")
        if not isinstance(joints, Sequence) or len(joints) != 7:
            raise ValueError(
                f"simulation sequence target {index} needs seven target joints"
            )
        if num_waypoints <= 0:
            raise ValueError(
                f"simulation sequence target {index} needs positive waypoints"
            )
        normalized_targets.append(
            {
                "phase": str(target.get("phase", f"segment_{index + 1}")),
                "pose": [float(value) for value in pose],
                "target_joints": [float(value) for value in joints],
                "num_waypoints": num_waypoints,
            }
        )
    return {
        "task": "sim_sequence",
        "arm": 0 if arm in {"left", "0"} else 1,
        "joints_left": [float(value) for value in left_joints_deg],
        "joints_right": [float(value) for value in right_joints_deg],
        "targets": normalized_targets,
        "recording": bool(recording),
    }


def require_response_fields(response: Any, fields: Sequence[str], *, service: str) -> Mapping[str, Any]:
    if not isinstance(response, Mapping):
        raise ProtocolError(f"{service} response must be a JSON object")
    missing = [field for field in fields if field not in response]
    if missing:
        raise ProtocolError(f"{service} response omitted fields: {', '.join(missing)}")
    return response


@dataclass(frozen=True)
class LegacyApiPorts:
    detection: int = 65444
    segmentation: int = 65471
    simulation: int = 65078
    grasp: int = 65432


class LegacyApiGateway:
    """Typed routing to the four current 4090 services, without hardware access."""

    def __init__(
        self,
        host: str,
        *,
        ports: LegacyApiPorts | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        selected = ports or LegacyApiPorts()
        self.detection = LengthPrefixedJsonClient(host, selected.detection, timeout_s=timeout_s)
        self.segmentation = LengthPrefixedJsonClient(host, selected.segmentation, timeout_s=timeout_s)
        self.simulation = LengthPrefixedJsonClient(host, selected.simulation, timeout_s=timeout_s)
        self.grasp = LengthPrefixedJsonClient(host, selected.grasp, timeout_s=timeout_s)

    def open_vocabulary(self, **arguments: Any) -> Mapping[str, Any]:
        response = self.detection.request(build_open_vocabulary_request(**arguments))
        return require_response_fields(response, ("masks", "labels"), service="detection")

    def segment(self, **arguments: Any) -> Mapping[str, Any]:
        response = self.segmentation.request(build_segmentation_request(**arguments))
        return require_response_fields(response, ("masks",), service="segmentation")

    def propose_grasps(self, **arguments: Any) -> Mapping[str, Any]:
        response = require_response_fields(
            self.grasp.request(build_grasp_request(**arguments)),
            ("pose", "gdepth", "gwidth", "index"),
            service="grasp",
        )
        lengths = {len(response[field]) for field in ("pose", "gdepth", "gwidth", "index")}
        if len(lengths) != 1:
            raise ProtocolError("grasp response arrays have inconsistent lengths")
        return response

    def simulate(self, **arguments: Any) -> Mapping[str, Any]:
        request = build_simulation_request(**arguments)
        response = require_response_fields(
            self.simulation.request(request), ("path", "success"), service="simulation"
        )
        if request["recording"] and response["success"] and not response.get("video"):
            raise ProtocolError("successful simulation omitted mandatory review video")
        return response

    def simulate_sequence(self, **arguments: Any) -> Mapping[str, Any]:
        request = build_simulation_sequence_request(**arguments)
        response = require_response_fields(
            self.simulation.request(request),
            ("path", "segments", "success"),
            service="simulation sequence",
        )
        if response["success"] and len(response["segments"]) != len(request["targets"]):
            raise ProtocolError("simulation sequence returned the wrong segment count")
        if request["recording"] and response["success"] and not response.get("video"):
            raise ProtocolError(
                "successful simulation sequence omitted mandatory review video"
            )
        return response
