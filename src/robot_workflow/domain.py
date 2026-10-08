"""Immutable domain models shared by perception, planning, approval, and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import time
from typing import Any, Iterable, Mapping
from uuid import uuid4

from .errors import SnapshotValidationError, ValidationError


JOINT_COUNT = 7


def _finite(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValidationError(f"{name} must contain only finite values")
    return result


def _sha256_hex(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


class CameraName(str, Enum):
    LEFT = "left"
    HEAD = "head"
    RIGHT = "right"


class MotionPhase(str, Enum):
    GRASP = "grasp"
    PLACE = "place"


class WorkflowState(str, Enum):
    IDLE = "idle"
    SENSING = "sensing"
    PERCEIVING = "perceiving"
    GRASP_PLANNING = "grasp_planning"
    AWAITING_GRASP_APPROVAL = "awaiting_grasp_approval"
    EXECUTING_GRASP = "executing_grasp"
    VERIFYING_GRASP = "verifying_grasp"
    PLACE_PLANNING = "place_planning"
    AWAITING_PLACE_APPROVAL = "awaiting_place_approval"
    EXECUTING_PLACE = "executing_place"
    VERIFYING_PLACE = "verifying_place"
    DONE = "done"
    ABORTED = "aborted"


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValidationError("camera dimensions must be positive")
        values = _finite((self.fx, self.fy, self.ppx, self.ppy), name="intrinsics")
        if values[0] <= 0 or values[1] <= 0:
            raise ValidationError("fx and fy must be positive")

    def canonical(self) -> dict[str, float | int]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "ppx": self.ppx,
            "ppy": self.ppy,
        }


@dataclass(frozen=True)
class RigidTransform:
    """world_from_camera transform, expressed in metres and quaternion xyzw."""

    translation_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        translation = _finite(self.translation_m, name="translation_m")
        quaternion = _finite(self.quaternion_xyzw, name="quaternion_xyzw")
        if len(translation) != 3 or len(quaternion) != 4:
            raise ValidationError("transform must have 3 translation and 4 quaternion values")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if not 0.999 <= norm <= 1.001:
            raise ValidationError("quaternion must be normalized")

    @staticmethod
    def identity() -> "RigidTransform":
        return RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def canonical(self) -> dict[str, list[float]]:
        return {
            "translation_m": list(self.translation_m),
            "quaternion_xyzw": list(self.quaternion_xyzw),
        }


@dataclass(frozen=True)
class CameraFrame:
    name: CameraName
    timestamp_ns: int
    intrinsics: Intrinsics | None
    rgb_encoding: str
    rgb_bytes: bytes = field(repr=False)
    depth_encoding: str | None = None
    depth_bytes: bytes | None = field(default=None, repr=False)
    world_from_camera: RigidTransform | None = None
    serial_number: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ns <= 0:
            raise ValidationError("camera timestamp must be positive")
        if not self.rgb_encoding or not self.rgb_bytes:
            raise ValidationError(f"{self.name.value} camera requires an RGB frame")
        if (self.depth_encoding is None) != (self.depth_bytes is None):
            raise ValidationError("depth_encoding and depth_bytes must be supplied together")

    @property
    def rgb_sha256(self) -> str:
        return hashlib.sha256(self.rgb_bytes).hexdigest()

    @property
    def depth_sha256(self) -> str | None:
        if self.depth_bytes is None:
            return None
        return hashlib.sha256(self.depth_bytes).hexdigest()

    def canonical(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "timestamp_ns": self.timestamp_ns,
            "intrinsics": self.intrinsics.canonical() if self.intrinsics else None,
            "rgb_encoding": self.rgb_encoding,
            "rgb_sha256": self.rgb_sha256,
            "depth_encoding": self.depth_encoding,
            "depth_sha256": self.depth_sha256,
            "world_from_camera": (
                self.world_from_camera.canonical() if self.world_from_camera else None
            ),
            "serial_number": self.serial_number,
        }


@dataclass(frozen=True)
class RobotState:
    left_joints_deg: tuple[float, ...]
    right_joints_deg: tuple[float, ...]
    left_gripper_open: bool
    right_gripper_open: bool

    def __post_init__(self) -> None:
        left = _finite(self.left_joints_deg, name="left_joints_deg")
        right = _finite(self.right_joints_deg, name="right_joints_deg")
        if len(left) != JOINT_COUNT or len(right) != JOINT_COUNT:
            raise ValidationError(f"each arm must contain {JOINT_COUNT} joint values")

    def joints_for(self, arm: str) -> tuple[float, ...]:
        if arm == "left":
            return self.left_joints_deg
        if arm == "right":
            return self.right_joints_deg
        raise ValidationError(f"unknown arm: {arm}")

    def gripper_open_for(self, arm: str) -> bool:
        if arm == "left":
            return self.left_gripper_open
        if arm == "right":
            return self.right_gripper_open
        raise ValidationError(f"unknown arm: {arm}")

    def canonical(self) -> dict[str, Any]:
        return {
            "left_joints_deg": list(self.left_joints_deg),
            "right_joints_deg": list(self.right_joints_deg),
            "left_gripper_open": self.left_gripper_open,
            "right_gripper_open": self.right_gripper_open,
        }


@dataclass(frozen=True)
class SceneSnapshot:
    scene_id: str
    captured_at_ns: int
    frames: tuple[CameraFrame, ...]
    robot_state: RobotState

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise ValidationError("scene_id is required")
        if self.captured_at_ns <= 0:
            raise ValidationError("captured_at_ns must be positive")
        names = tuple(frame.name for frame in self.frames)
        required = {CameraName.LEFT, CameraName.HEAD, CameraName.RIGHT}
        if len(names) != len(set(names)):
            raise SnapshotValidationError("camera names must be unique")
        if set(names) != required:
            missing = sorted(name.value for name in required - set(names))
            extra = sorted(name.value for name in set(names) - required)
            raise SnapshotValidationError(f"three-camera snapshot mismatch: missing={missing}, extra={extra}")

    def frame(self, name: CameraName) -> CameraFrame:
        for frame in self.frames:
            if frame.name == name:
                return frame
        raise SnapshotValidationError(f"missing {name.value} camera")

    def validate_freshness(self, *, now_ns: int, max_age_ms: int, max_skew_ms: int) -> None:
        timestamps = [frame.timestamp_ns for frame in self.frames]
        if max(timestamps) > now_ns + max_skew_ms * 1_000_000:
            raise SnapshotValidationError("camera snapshot timestamp is in the future")
        if now_ns - min(timestamps) > max_age_ms * 1_000_000:
            raise SnapshotValidationError("camera snapshot is stale")
        if max(timestamps) - min(timestamps) > max_skew_ms * 1_000_000:
            raise SnapshotValidationError("three camera frames are not sufficiently synchronized")

    def require_geometry(self, camera_names: Iterable[CameraName]) -> None:
        for name in camera_names:
            frame = self.frame(name)
            missing: list[str] = []
            if frame.depth_bytes is None:
                missing.append("depth")
            if frame.intrinsics is None:
                missing.append("intrinsics")
            if frame.world_from_camera is None:
                missing.append("world_from_camera")
            if missing:
                raise SnapshotValidationError(
                    f"{name.value} camera is not 3D-fusion ready: missing {', '.join(missing)}"
                )

    def canonical(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "captured_at_ns": self.captured_at_ns,
            "frames": [frame.canonical() for frame in sorted(self.frames, key=lambda item: item.name.value)],
            "robot_state": self.robot_state.canonical(),
        }


@dataclass(frozen=True)
class TaskRequest:
    target_query: str
    receptacle_query: str
    task_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not self.target_query.strip() or not self.receptacle_query.strip():
            raise ValidationError("target and receptacle queries are required")


@dataclass(frozen=True)
class DetectedObject:
    object_id: str
    label: str
    confidence: float
    source_cameras: tuple[CameraName, ...]
    centroid_world_m: tuple[float, float, float]
    bbox_xyxy_by_camera: tuple[tuple[CameraName, tuple[int, int, int, int]], ...] = ()
    mask_reference_by_camera: tuple[tuple[CameraName, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.object_id or not self.label:
            raise ValidationError("detected object id and label are required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("object confidence must be in [0, 1]")
        centroid = _finite(self.centroid_world_m, name="centroid_world_m")
        if len(centroid) != 3:
            raise ValidationError("object centroid must contain three values")
        if not self.source_cameras:
            raise ValidationError("detected object must identify at least one source camera")


@dataclass(frozen=True)
class SceneAnalysis:
    scene_id: str
    target: DetectedObject
    receptacle: DetectedObject
    obstacle_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    arm: str
    pose_world_m_rad: tuple[float, float, float, float, float, float]
    score: float
    width_m: float
    depth_m: float
    collision_free: bool
    minimum_clearance_m: float

    def __post_init__(self) -> None:
        if self.arm not in {"left", "right"}:
            raise ValidationError("grasp candidate arm must be left or right")
        pose = _finite(self.pose_world_m_rad, name="pose_world_m_rad")
        if len(pose) != 6:
            raise ValidationError("grasp pose must contain six values")
        if not 0.0 <= self.score <= 1.0:
            raise ValidationError("grasp score must be in [0, 1]")
        if self.width_m <= 0 or self.depth_m < 0 or self.minimum_clearance_m < 0:
            raise ValidationError("grasp dimensions and clearance must be non-negative")


@dataclass(frozen=True)
class TrajectoryPoint:
    time_from_start_s: float
    positions_deg: tuple[float, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_from_start_s) or self.time_from_start_s <= 0:
            raise ValidationError("trajectory time must be positive and finite")
        positions = _finite(self.positions_deg, name="positions_deg")
        if len(positions) != JOINT_COUNT:
            raise ValidationError(f"trajectory point must contain {JOINT_COUNT} joint values")

    def canonical(self) -> dict[str, Any]:
        return {
            "time_from_start_s": self.time_from_start_s,
            "positions_deg": list(self.positions_deg),
        }


@dataclass(frozen=True)
class PreviewArtifact:
    uri: str
    sha256: str
    media_type: str
    duration_ms: int
    frame_count: int

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValidationError("preview URI is required")
        if not _sha256_hex(self.sha256):
            raise ValidationError("preview sha256 must be a 64-character hex digest")
        if self.media_type != "video/mp4":
            raise ValidationError("preview must be an MP4 video")
        if self.duration_ms <= 0 or self.frame_count <= 0:
            raise ValidationError("preview duration and frame count must be positive")

    def canonical(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "duration_ms": self.duration_ms,
            "frame_count": self.frame_count,
        }


@dataclass(frozen=True)
class CollisionReport:
    collision_free: bool
    minimum_clearance_m: float
    checked_scene_geometry: tuple[str, ...]
    attached_object_id: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_clearance_m) or self.minimum_clearance_m < 0:
            raise ValidationError("minimum clearance must be finite and non-negative")

    def canonical(self) -> dict[str, Any]:
        return {
            "collision_free": self.collision_free,
            "minimum_clearance_m": self.minimum_clearance_m,
            "checked_scene_geometry": list(self.checked_scene_geometry),
            "attached_object_id": self.attached_object_id,
        }


@dataclass(frozen=True)
class MotionPlan:
    plan_id: str
    task_id: str
    scene_id: str
    phase: MotionPhase
    arm: str
    start_state: RobotState
    waypoints: tuple[TrajectoryPoint, ...]
    preview: PreviewArtifact
    collision_report: CollisionReport
    created_at_s: float
    expires_at_s: float
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.plan_id or not self.task_id or not self.scene_id:
            raise ValidationError("plan, task, and scene identifiers are required")
        if self.arm not in {"left", "right"}:
            raise ValidationError("plan arm must be left or right")
        if not self.waypoints:
            raise ValidationError("motion plan requires at least one waypoint")
        if not math.isfinite(self.created_at_s) or not math.isfinite(self.expires_at_s):
            raise ValidationError("plan timestamps must be finite")
        if self.expires_at_s <= self.created_at_s:
            raise ValidationError("plan expiry must be after creation")
        previous = 0.0
        for point in self.waypoints:
            if point.time_from_start_s <= previous:
                raise ValidationError("trajectory timestamps must be strictly increasing")
            previous = point.time_from_start_s

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "scene_id": self.scene_id,
            "phase": self.phase.value,
            "arm": self.arm,
            "start_state": self.start_state.canonical(),
            "waypoints": [point.canonical() for point in self.waypoints],
            "preview": self.preview.canonical(),
            "collision_report": self.collision_report.canonical(),
            "created_at_s": self.created_at_s,
            "expires_at_s": self.expires_at_s,
            "metadata": [list(item) for item in self.metadata],
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.canonical_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def new_id() -> str:
        return str(uuid4())


@dataclass(frozen=True)
class PendingReview:
    plan_id: str
    plan_digest: str
    phase: MotionPhase
    preview: PreviewArtifact
    expires_at_s: float

    @classmethod
    def from_plan(cls, plan: MotionPlan) -> "PendingReview":
        return cls(
            plan_id=plan.plan_id,
            plan_digest=plan.digest,
            phase=plan.phase,
            preview=plan.preview,
            expires_at_s=plan.expires_at_s,
        )


@dataclass(frozen=True)
class ExecutionResult:
    plan_id: str
    success: bool
    final_state: RobotState
    message: str


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    evidence: tuple[str, ...]
    message: str


def wall_time_s() -> float:
    return time.time()
