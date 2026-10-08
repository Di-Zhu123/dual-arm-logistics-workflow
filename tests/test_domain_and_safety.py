from __future__ import annotations

from dataclasses import replace
import time
import unittest

from robot_workflow.approval import ApprovalAuthority, ApprovalToken
from robot_workflow.domain import (
    CameraFrame,
    CameraName,
    CollisionReport,
    MotionPhase,
    MotionPlan,
    PreviewArtifact,
    RobotState,
    SceneSnapshot,
    TrajectoryPoint,
)
from robot_workflow.errors import ApprovalError, SafetyViolation, SnapshotValidationError
from robot_workflow.offline import SimulationWorld, SyntheticThreeCameraSceneProvider
from robot_workflow.safety import SafetyPolicy


def make_plan(*, phase: MotionPhase = MotionPhase.GRASP) -> MotionPlan:
    now = time.time()
    state = RobotState((0.0,) * 7, (0.0,) * 7, True, True)
    return MotionPlan(
        plan_id="plan-1",
        task_id="task-1",
        scene_id="scene-1",
        phase=phase,
        arm="left",
        start_state=state,
        waypoints=(TrajectoryPoint(1.0, (2.0,) * 7), TrajectoryPoint(2.0, (4.0,) * 7)),
        preview=PreviewArtifact("memory://preview.mp4", "a" * 64, "video/mp4", 1000, 30),
        collision_report=CollisionReport(
            True,
            0.04,
            ("left_arm", "right_arm", "table", "receptacle", "obstacles"),
            attached_object_id="target-1" if phase == MotionPhase.PLACE else None,
        ),
        created_at_s=now,
        expires_at_s=now + 60,
    )


class SnapshotTests(unittest.TestCase):
    def test_snapshot_requires_exactly_three_named_cameras(self) -> None:
        provider = SyntheticThreeCameraSceneProvider(SimulationWorld())
        valid = provider.capture()
        with self.assertRaisesRegex(SnapshotValidationError, "mismatch"):
            SceneSnapshot(
                valid.scene_id,
                valid.captured_at_ns,
                tuple(frame for frame in valid.frames if frame.name != CameraName.HEAD),
                valid.robot_state,
            )

    def test_duplicate_camera_is_rejected(self) -> None:
        provider = SyntheticThreeCameraSceneProvider(SimulationWorld())
        valid = provider.capture()
        duplicate = (valid.frames[0], valid.frames[0], valid.frames[2])
        with self.assertRaisesRegex(SnapshotValidationError, "unique"):
            SceneSnapshot(valid.scene_id, valid.captured_at_ns, duplicate, valid.robot_state)

    def test_missing_head_geometry_is_rejected(self) -> None:
        provider = SyntheticThreeCameraSceneProvider(SimulationWorld())
        valid = provider.capture()
        frames = tuple(
            replace(frame, world_from_camera=None)
            if frame.name == CameraName.HEAD
            else frame
            for frame in valid.frames
        )
        snapshot = replace(valid, frames=frames)
        with self.assertRaisesRegex(SnapshotValidationError, "head.*world_from_camera"):
            SafetyPolicy().validate_snapshot(snapshot)

    def test_stale_and_skewed_snapshot_are_rejected(self) -> None:
        provider = SyntheticThreeCameraSceneProvider(SimulationWorld())
        valid = provider.capture()
        old_frames = tuple(replace(frame, timestamp_ns=1) for frame in valid.frames)
        with self.assertRaisesRegex(SnapshotValidationError, "stale"):
            SafetyPolicy().validate_snapshot(replace(valid, frames=old_frames))

        now = time.time_ns()
        skewed = tuple(
            replace(frame, timestamp_ns=now - (300_000_000 if index == 0 else 0))
            for index, frame in enumerate(valid.frames)
        )
        with self.assertRaisesRegex(SnapshotValidationError, "synchronized"):
            SafetyPolicy().validate_snapshot(replace(valid, frames=skewed), now_ns=now)


class MotionPlanAndSafetyTests(unittest.TestCase):
    def test_plan_digest_detects_trajectory_tampering(self) -> None:
        plan = make_plan()
        changed = replace(
            plan,
            waypoints=(TrajectoryPoint(1.0, (3.0,) * 7), plan.waypoints[1]),
        )
        self.assertNotEqual(plan.digest, changed.digest)

    def test_collision_clearance_and_geometry_are_enforced(self) -> None:
        plan = make_plan()
        policy = SafetyPolicy()
        policy.validate_reviewable(plan, expected_phase=MotionPhase.GRASP)

        collision = replace(plan, collision_report=replace(plan.collision_report, collision_free=False))
        with self.assertRaisesRegex(SafetyViolation, "collision"):
            policy.validate_reviewable(collision, expected_phase=MotionPhase.GRASP)

        low_clearance = replace(
            plan,
            collision_report=replace(plan.collision_report, minimum_clearance_m=0.001),
        )
        with self.assertRaisesRegex(SafetyViolation, "clearance"):
            policy.validate_reviewable(low_clearance, expected_phase=MotionPhase.GRASP)

        missing_geometry = replace(
            plan,
            collision_report=replace(plan.collision_report, checked_scene_geometry=("table",)),
        )
        with self.assertRaisesRegex(SafetyViolation, "omitted"):
            policy.validate_reviewable(missing_geometry, expected_phase=MotionPhase.GRASP)

        excessive_lifetime = replace(plan, expires_at_s=plan.created_at_s + 301.0)
        with self.assertRaisesRegex(SafetyViolation, "lifetime"):
            policy.validate_reviewable(excessive_lifetime, expected_phase=MotionPhase.GRASP)

    def test_place_requires_attached_object(self) -> None:
        plan = make_plan(phase=MotionPhase.PLACE)
        invalid = replace(
            plan,
            collision_report=replace(plan.collision_report, attached_object_id=None),
        )
        with self.assertRaisesRegex(SafetyViolation, "attached"):
            SafetyPolicy().validate_reviewable(invalid, expected_phase=MotionPhase.PLACE)

    def test_velocity_joint_limit_and_start_drift_are_enforced(self) -> None:
        plan = make_plan()
        fast = replace(plan, waypoints=(TrajectoryPoint(0.01, (2.0,) * 7),))
        with self.assertRaisesRegex(SafetyViolation, "velocity"):
            SafetyPolicy().validate_reviewable(fast, expected_phase=MotionPhase.GRASP)

        over_limit = replace(plan, waypoints=(TrajectoryPoint(10.0, (181.0,) * 7),))
        with self.assertRaisesRegex(SafetyViolation, "position"):
            SafetyPolicy().validate_reviewable(over_limit, expected_phase=MotionPhase.GRASP)

        drifted = replace(plan.start_state, left_joints_deg=(1.0,) + (0.0,) * 6)
        with self.assertRaisesRegex(SafetyViolation, "drifted"):
            SafetyPolicy().validate_pre_execution(
                plan,
                current_state=drifted,
                scene_is_current=True,
            )

        inactive_arm_drift = replace(plan.start_state, right_joints_deg=(1.0,) + (0.0,) * 6)
        with self.assertRaisesRegex(SafetyViolation, "right.*drifted"):
            SafetyPolicy().validate_pre_execution(
                plan,
                current_state=inactive_arm_drift,
                scene_is_current=True,
            )

        gripper_changed = replace(plan.start_state, left_gripper_open=False)
        with self.assertRaisesRegex(SafetyViolation, "gripper state changed"):
            SafetyPolicy().validate_pre_execution(
                plan,
                current_state=gripper_changed,
                scene_is_current=True,
            )


class ApprovalTests(unittest.TestCase):
    def test_wrong_digest_is_not_approved(self) -> None:
        plan = make_plan()
        with self.assertRaisesRegex(ApprovalError, "digest"):
            ApprovalAuthority().approve(plan, presented_digest="0" * 64, reviewer="operator")

    def test_token_is_bound_and_one_time(self) -> None:
        plan = make_plan()
        authority = ApprovalAuthority()
        token = authority.approve(plan, presented_digest=plan.digest, reviewer="operator")
        authority.consume(plan, token)
        self.assertTrue(authority.is_consumed(plan.plan_id))
        with self.assertRaisesRegex(ApprovalError, "consumed"):
            authority.consume(plan, token)

    def test_invalid_token_and_expiry_are_rejected(self) -> None:
        plan = make_plan()
        authority = ApprovalAuthority()
        token = authority.approve(plan, presented_digest=plan.digest, reviewer="operator")
        with self.assertRaisesRegex(ApprovalError, "invalid"):
            authority.consume(plan, ApprovalToken(plan.plan_id, token.value + "x"))
        with self.assertRaisesRegex(ApprovalError, "expired"):
            authority.consume(plan, token, now_s=plan.expires_at_s)

    def test_plan_content_change_invalidates_existing_approval(self) -> None:
        plan = make_plan()
        authority = ApprovalAuthority()
        token = authority.approve(plan, presented_digest=plan.digest, reviewer="operator")
        changed = replace(plan, waypoints=(TrajectoryPoint(1.0, (3.0,) * 7),))
        with self.assertRaisesRegex(ApprovalError, "content changed"):
            authority.consume(changed, token)

    def test_consumed_plan_cannot_be_reapproved(self) -> None:
        plan = make_plan()
        authority = ApprovalAuthority()
        token = authority.approve(plan, presented_digest=plan.digest, reviewer="operator")
        authority.consume(plan, token)
        with self.assertRaisesRegex(ApprovalError, "new plan"):
            authority.approve(plan, presented_digest=plan.digest, reviewer="operator")


if __name__ == "__main__":
    unittest.main()
