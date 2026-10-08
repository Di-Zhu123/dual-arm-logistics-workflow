"""Dependency-free simulator and fixture reader for offline workflow tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

from .domain import (
    CameraFrame,
    CameraName,
    CollisionReport,
    DetectedObject,
    ExecutionResult,
    GraspCandidate,
    Intrinsics,
    MotionPhase,
    MotionPlan,
    PreviewArtifact,
    RigidTransform,
    RobotState,
    SceneAnalysis,
    SceneSnapshot,
    TaskRequest,
    TrajectoryPoint,
    VerificationResult,
)
from .errors import ServiceError
from .protocols import ExecutorMode


ZERO_STATE = RobotState(
    left_joints_deg=(0.0,) * 7,
    right_joints_deg=(0.0,) * 7,
    left_gripper_open=True,
    right_gripper_open=True,
)


@dataclass
class SimulationWorld:
    robot_state: RobotState = ZERO_STATE
    revision: int = 0
    held_object_id: str | None = None
    placed_object_id: str | None = None

    @property
    def scene_id(self) -> str:
        return f"simulation-scene-{self.revision}"

    def mutate(self) -> None:
        self.revision += 1


class SyntheticThreeCameraSceneProvider:
    def __init__(self, world: SimulationWorld) -> None:
        self.world = world

    def capture(self) -> SceneSnapshot:
        timestamp = time.time_ns()
        intrinsics = Intrinsics(640, 480, 600.0, 600.0, 320.0, 240.0)
        frames = tuple(
            CameraFrame(
                name=name,
                timestamp_ns=timestamp,
                intrinsics=intrinsics,
                rgb_encoding="image/jpeg",
                rgb_bytes=b"offline-rgb-" + name.value.encode("ascii"),
                depth_encoding="uint16-mm",
                depth_bytes=b"offline-depth-" + name.value.encode("ascii"),
                world_from_camera=RigidTransform.identity(),
                serial_number=f"offline-{name.value}",
            )
            for name in (CameraName.LEFT, CameraName.HEAD, CameraName.RIGHT)
        )
        return SceneSnapshot(
            scene_id=self.world.scene_id,
            captured_at_ns=timestamp,
            frames=frames,
            robot_state=self.world.robot_state,
        )

    def scene_is_current(self, scene_id: str) -> bool:
        return scene_id == self.world.scene_id


class FixtureReplaySceneProvider:
    """Read the copied three-RGB fixture without pretending geometry exists."""

    def __init__(self, frame_directory: str | Path) -> None:
        self.frame_directory = Path(frame_directory)
        self._scene_id = f"fixture-{self.frame_directory.name}"

    def capture(self) -> SceneSnapshot:
        timestamp = time.time_ns()
        frames = tuple(
            CameraFrame(
                name=name,
                timestamp_ns=timestamp,
                intrinsics=None,
                rgb_encoding="image/jpeg",
                rgb_bytes=(self.frame_directory / f"{name.value}.jpg").read_bytes(),
            )
            for name in (CameraName.LEFT, CameraName.HEAD, CameraName.RIGHT)
        )
        raw_state = json.loads((self.frame_directory / "state.json").read_text("utf-8"))
        state = RobotState(
            left_joints_deg=tuple(raw_state["left_joints"]),
            right_joints_deg=tuple(raw_state["right_joints"]),
            left_gripper_open=bool(raw_state["left_gripper"]),
            right_gripper_open=bool(raw_state["right_gripper"]),
        )
        return SceneSnapshot(self._scene_id, timestamp, frames, state)

    def scene_is_current(self, scene_id: str) -> bool:
        return scene_id == self._scene_id


class OfflinePerception:
    def analyze(self, snapshot: SceneSnapshot, task: TaskRequest) -> SceneAnalysis:
        all_cameras = (CameraName.LEFT, CameraName.HEAD, CameraName.RIGHT)
        target = DetectedObject(
            object_id="target-1",
            label=task.target_query,
            confidence=0.96,
            source_cameras=all_cameras,
            centroid_world_m=(0.42, 0.10, 0.08),
        )
        receptacle = DetectedObject(
            object_id="receptacle-1",
            label=task.receptacle_query,
            confidence=0.98,
            source_cameras=all_cameras,
            centroid_world_m=(0.48, -0.16, 0.10),
        )
        return SceneAnalysis(snapshot.scene_id, target, receptacle, ("obstacle-1",))


class OfflineGraspService:
    def __init__(self, candidates: Sequence[GraspCandidate] | None = None) -> None:
        self._candidates = candidates

    def propose(
        self,
        snapshot: SceneSnapshot,
        analysis: SceneAnalysis,
    ) -> Sequence[GraspCandidate]:
        if self._candidates is not None:
            return self._candidates
        return (
            GraspCandidate(
                candidate_id="grasp-left-1",
                arm="left",
                pose_world_m_rad=(0.42, 0.10, 0.12, 3.14, 0.0, 0.0),
                score=0.94,
                width_m=0.06,
                depth_m=0.04,
                collision_free=True,
                minimum_clearance_m=0.04,
            ),
        )


class OfflineMotionPlanner:
    checked_geometry = ("left_arm", "right_arm", "table", "receptacle", "obstacles")

    @staticmethod
    def _preview(phase: MotionPhase, scene_id: str) -> PreviewArtifact:
        content = f"offline-mp4:{phase.value}:{scene_id}".encode("utf-8")
        return PreviewArtifact(
            uri=f"memory://{scene_id}/{phase.value}.mp4",
            sha256=hashlib.sha256(content).hexdigest(),
            media_type="video/mp4",
            duration_ms=2_000,
            frame_count=60,
        )

    @staticmethod
    def _waypoints(start: tuple[float, ...]) -> tuple[TrajectoryPoint, ...]:
        return (
            TrajectoryPoint(1.0, tuple(value + 2.0 for value in start)),
            TrajectoryPoint(2.0, tuple(value + 4.0 for value in start)),
        )

    def plan_grasp(
        self,
        task: TaskRequest,
        snapshot: SceneSnapshot,
        analysis: SceneAnalysis,
        candidates: Sequence[GraspCandidate],
    ) -> MotionPlan:
        if not candidates:
            raise ServiceError("offline planner received no candidates")
        selected = candidates[0]
        now = time.time()
        return MotionPlan(
            plan_id=MotionPlan.new_id(),
            task_id=task.task_id,
            scene_id=snapshot.scene_id,
            phase=MotionPhase.GRASP,
            arm=selected.arm,
            start_state=snapshot.robot_state,
            waypoints=self._waypoints(snapshot.robot_state.joints_for(selected.arm)),
            preview=self._preview(MotionPhase.GRASP, snapshot.scene_id),
            collision_report=CollisionReport(True, 0.04, self.checked_geometry),
            created_at_s=now,
            expires_at_s=now + 300.0,
            metadata=(("candidate_id", selected.candidate_id), ("target_id", analysis.target.object_id)),
        )

    def plan_place(
        self,
        task: TaskRequest,
        snapshot: SceneSnapshot,
        analysis: SceneAnalysis,
        selected_grasp: GraspCandidate,
    ) -> MotionPlan:
        now = time.time()
        return MotionPlan(
            plan_id=MotionPlan.new_id(),
            task_id=task.task_id,
            scene_id=snapshot.scene_id,
            phase=MotionPhase.PLACE,
            arm=selected_grasp.arm,
            start_state=snapshot.robot_state,
            waypoints=self._waypoints(snapshot.robot_state.joints_for(selected_grasp.arm)),
            preview=self._preview(MotionPhase.PLACE, snapshot.scene_id),
            collision_report=CollisionReport(
                True,
                0.04,
                self.checked_geometry,
                attached_object_id=analysis.target.object_id,
            ),
            created_at_s=now,
            expires_at_s=now + 300.0,
            metadata=(("candidate_id", selected_grasp.candidate_id), ("target_id", analysis.target.object_id)),
        )


class SimulatedExecutor:
    mode = ExecutorMode.SIMULATION

    def __init__(self, world: SimulationWorld) -> None:
        self.world = world
        self.history: list[MotionPlan] = []
        self.fail_next = False

    def current_state(self) -> RobotState:
        return self.world.robot_state

    @staticmethod
    def _metadata(plan: MotionPlan, key: str) -> str | None:
        return dict(plan.metadata).get(key)

    def execute(self, plan: MotionPlan) -> ExecutionResult:
        if self.fail_next:
            self.fail_next = False
            return ExecutionResult(plan.plan_id, False, self.world.robot_state, "injected failure")
        final_joints = plan.waypoints[-1].positions_deg
        if plan.arm == "left":
            state = replace(
                self.world.robot_state,
                left_joints_deg=final_joints,
                left_gripper_open=plan.phase == MotionPhase.PLACE,
            )
        else:
            state = replace(
                self.world.robot_state,
                right_joints_deg=final_joints,
                right_gripper_open=plan.phase == MotionPhase.PLACE,
            )
        target_id = self._metadata(plan, "target_id")
        if plan.phase == MotionPhase.GRASP:
            self.world.held_object_id = target_id
        else:
            self.world.placed_object_id = self.world.held_object_id
            self.world.held_object_id = None
        self.world.robot_state = state
        self.world.mutate()
        self.history.append(plan)
        return ExecutionResult(plan.plan_id, True, state, "simulated execution complete")


class OfflineVerifier:
    def __init__(self, world: SimulationWorld) -> None:
        self.world = world
        self.fail_grasp = False
        self.fail_place = False

    def verify_grasp(
        self,
        task: TaskRequest,
        before: SceneSnapshot,
        execution: ExecutionResult,
    ) -> VerificationResult:
        success = not self.fail_grasp and self.world.held_object_id is not None
        return VerificationResult(
            success,
            ("simulated gripper state", "simulated post-motion scene"),
            "target held" if success else "target not held",
        )

    def verify_place(
        self,
        task: TaskRequest,
        before: SceneSnapshot,
        execution: ExecutionResult,
    ) -> VerificationResult:
        success = not self.fail_place and self.world.placed_object_id is not None
        return VerificationResult(
            success,
            ("simulated containment test", "simulated empty gripper"),
            "target inside receptacle" if success else "placement not confirmed",
        )
