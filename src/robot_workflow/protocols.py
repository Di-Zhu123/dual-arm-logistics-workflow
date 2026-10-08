"""Dependency-injection ports. Real hardware adapters are intentionally absent."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, Sequence

from .domain import (
    ExecutionResult,
    GraspCandidate,
    MotionPlan,
    RobotState,
    SceneAnalysis,
    SceneSnapshot,
    TaskRequest,
    VerificationResult,
)


class ExecutorMode(str, Enum):
    SIMULATION = "simulation"
    HARDWARE = "hardware"


class SceneProvider(Protocol):
    def capture(self) -> SceneSnapshot:
        """Capture a synchronized left/head/right snapshot and robot state."""

    def scene_is_current(self, scene_id: str) -> bool:
        """Return whether the physical/simulated scene still matches scene_id."""


class PerceptionService(Protocol):
    def analyze(self, snapshot: SceneSnapshot, task: TaskRequest) -> SceneAnalysis:
        """Detect, segment, and geometrically localize target and receptacle."""


class GraspService(Protocol):
    def propose(
        self,
        snapshot: SceneSnapshot,
        analysis: SceneAnalysis,
    ) -> Sequence[GraspCandidate]:
        """Return collision-filtered, task-aware grasp candidates."""


class MotionPlanner(Protocol):
    def plan_grasp(
        self,
        task: TaskRequest,
        snapshot: SceneSnapshot,
        analysis: SceneAnalysis,
        candidates: Sequence[GraspCandidate],
    ) -> MotionPlan:
        """Plan pre-grasp, approach, close, lift, and retreat."""

    def plan_place(
        self,
        task: TaskRequest,
        snapshot: SceneSnapshot,
        analysis: SceneAnalysis,
        selected_grasp: GraspCandidate,
    ) -> MotionPlan:
        """Plan transit, insertion, release, and retreat with an attached object."""


class Executor(Protocol):
    mode: ExecutorMode

    def current_state(self) -> RobotState:
        """Return current joints and gripper state without changing hardware."""

    def execute(self, plan: MotionPlan) -> ExecutionResult:
        """Execute exactly the supplied immutable plan."""


class Verifier(Protocol):
    def verify_grasp(
        self,
        task: TaskRequest,
        before: SceneSnapshot,
        execution: ExecutionResult,
    ) -> VerificationResult:
        """Verify the target is held using independent evidence."""

    def verify_place(
        self,
        task: TaskRequest,
        before: SceneSnapshot,
        execution: ExecutionResult,
    ) -> VerificationResult:
        """Verify the target is inside the receptacle using independent evidence."""
