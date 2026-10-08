"""Mechanical and workflow invariants enforced independently of the LLM."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

from .domain import CameraName, MotionPhase, MotionPlan, RobotState, SceneSnapshot
from .errors import SafetyViolation


@dataclass(frozen=True)
class SafetyPolicy:
    required_geometry_cameras: tuple[CameraName, ...] = (
        CameraName.LEFT,
        CameraName.HEAD,
        CameraName.RIGHT,
    )
    maximum_snapshot_age_ms: int = 2_000
    maximum_camera_skew_ms: int = 250
    minimum_clearance_m: float = 0.015
    maximum_joint_velocity_deg_s: float = 30.0
    maximum_plan_lifetime_s: float = 300.0
    start_joint_tolerance_deg: float = 0.5
    joint_limits_deg: tuple[tuple[float, float], ...] = ((-180.0, 180.0),) * 7
    required_scene_geometry: tuple[str, ...] = (
        "left_arm",
        "right_arm",
        "table",
        "receptacle",
        "obstacles",
    )

    def __post_init__(self) -> None:
        if self.maximum_snapshot_age_ms <= 0 or self.maximum_camera_skew_ms < 0:
            raise SafetyViolation("snapshot age/skew policy must be non-negative")
        if (
            self.minimum_clearance_m < 0
            or self.maximum_joint_velocity_deg_s <= 0
            or self.maximum_plan_lifetime_s <= 0
        ):
            raise SafetyViolation("clearance and velocity policies must be positive")
        if self.start_joint_tolerance_deg < 0 or len(self.joint_limits_deg) != 7:
            raise SafetyViolation("joint policy is invalid")

    def validate_snapshot(self, snapshot: SceneSnapshot, *, now_ns: int | None = None) -> None:
        current_ns = time.time_ns() if now_ns is None else now_ns
        snapshot.validate_freshness(
            now_ns=current_ns,
            max_age_ms=self.maximum_snapshot_age_ms,
            max_skew_ms=self.maximum_camera_skew_ms,
        )
        snapshot.require_geometry(self.required_geometry_cameras)

    def validate_reviewable(self, plan: MotionPlan, *, expected_phase: MotionPhase) -> None:
        if plan.phase != expected_phase:
            raise SafetyViolation(
                f"expected a {expected_phase.value} plan, received {plan.phase.value}"
            )
        if plan.expires_at_s - plan.created_at_s > self.maximum_plan_lifetime_s:
            raise SafetyViolation("motion plan lifetime exceeds the configured maximum")
        report = plan.collision_report
        if not report.collision_free:
            raise SafetyViolation("planner reported a collision")
        if report.minimum_clearance_m < self.minimum_clearance_m:
            raise SafetyViolation("planned path does not meet minimum clearance")
        missing_geometry = set(self.required_scene_geometry) - set(report.checked_scene_geometry)
        if missing_geometry:
            raise SafetyViolation(
                "collision report omitted required geometry: " + ", ".join(sorted(missing_geometry))
            )
        if expected_phase == MotionPhase.PLACE and not report.attached_object_id:
            raise SafetyViolation("place planning must include the held object as attached geometry")

        active = plan.start_state.joints_for(plan.arm)
        previous_time = 0.0
        previous_positions = active
        for point in plan.waypoints:
            delta_t = point.time_from_start_s - previous_time
            if delta_t <= 0:
                raise SafetyViolation("trajectory timestamps are not strictly increasing")
            for index, position in enumerate(point.positions_deg):
                lower, upper = self.joint_limits_deg[index]
                if not lower <= position <= upper:
                    raise SafetyViolation(f"joint {index} exceeds configured position limits")
                velocity = abs(position - previous_positions[index]) / delta_t
                if velocity > self.maximum_joint_velocity_deg_s:
                    raise SafetyViolation(f"joint {index} exceeds configured velocity limit")
            previous_time = point.time_from_start_s
            previous_positions = point.positions_deg

    def validate_pre_execution(
        self,
        plan: MotionPlan,
        *,
        current_state: RobotState,
        scene_is_current: bool,
        now_s: float | None = None,
    ) -> None:
        now = time.time() if now_s is None else now_s
        if now >= plan.expires_at_s:
            raise SafetyViolation("motion plan has expired")
        if not scene_is_current:
            raise SafetyViolation("scene changed after planning")
        for arm in ("left", "right"):
            expected = plan.start_state.joints_for(arm)
            actual = current_state.joints_for(arm)
            drift = max(abs(left - right) for left, right in zip(expected, actual))
            if drift > self.start_joint_tolerance_deg:
                raise SafetyViolation(
                    f"{arm} start joint state drifted by {drift:.3f} deg; replan is required"
                )
            if plan.start_state.gripper_open_for(arm) != current_state.gripper_open_for(arm):
                raise SafetyViolation(f"{arm} gripper state changed after planning")
