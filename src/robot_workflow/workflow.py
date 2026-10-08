"""Deterministic, approval-gated orchestration for grasp then place."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from .approval import ApprovalAuthority, ApprovalToken
from .domain import (
    GraspCandidate,
    MotionPhase,
    MotionPlan,
    PendingReview,
    SceneAnalysis,
    SceneSnapshot,
    TaskRequest,
    VerificationResult,
    WorkflowState,
)
from .errors import SafetyViolation, ServiceError, TransitionError, WorkflowError
from .protocols import (
    Executor,
    ExecutorMode,
    GraspService,
    MotionPlanner,
    PerceptionService,
    SceneProvider,
    Verifier,
)
from .safety import SafetyPolicy


@dataclass(frozen=True)
class WorkflowConfig:
    """Hardware is denied by default and cannot be enabled accidentally."""

    allow_hardware_execution: bool = False


@dataclass(frozen=True)
class AuditEvent:
    sequence: int
    occurred_at_s: float
    previous_state: WorkflowState
    state: WorkflowState
    detail: str


@dataclass(frozen=True)
class WorkflowOutcome:
    task_id: str
    state: WorkflowState
    grasp_verification: VerificationResult
    place_verification: VerificationResult


class PickPlaceWorkflow:
    """A single-task state machine with a separate approval for each motion.

    This object never imports a robot SDK.  All side effects are behind injected
    interfaces, which lets exactly the same state machine run against a simulator
    before a hardware adapter is even made available.
    """

    def __init__(
        self,
        *,
        scene_provider: SceneProvider,
        perception: PerceptionService,
        grasp_service: GraspService,
        planner: MotionPlanner,
        executor: Executor,
        verifier: Verifier,
        approvals: ApprovalAuthority | None = None,
        safety: SafetyPolicy | None = None,
        config: WorkflowConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._scene_provider = scene_provider
        self._perception = perception
        self._grasp_service = grasp_service
        self._planner = planner
        self._executor = executor
        self._verifier = verifier
        self._approvals = approvals or ApprovalAuthority()
        self._safety = safety or SafetyPolicy()
        self._config = config or WorkflowConfig()
        self._clock = clock

        if (
            self._executor.mode == ExecutorMode.HARDWARE
            and not self._config.allow_hardware_execution
        ):
            raise SafetyViolation(
                "hardware executor rejected: allow_hardware_execution is false"
            )

        self.state = WorkflowState.IDLE
        self.audit_log: list[AuditEvent] = []
        self._task: TaskRequest | None = None
        self._snapshot: SceneSnapshot | None = None
        self._analysis: SceneAnalysis | None = None
        self._selected_grasp: GraspCandidate | None = None
        self._pending_plan: MotionPlan | None = None
        self._grasp_verification: VerificationResult | None = None

    @property
    def pending_review(self) -> PendingReview | None:
        if self._pending_plan is None:
            return None
        return PendingReview.from_plan(self._pending_plan)

    def _transition(self, state: WorkflowState, detail: str) -> None:
        previous = self.state
        self.state = state
        self.audit_log.append(
            AuditEvent(
                sequence=len(self.audit_log) + 1,
                occurred_at_s=self._clock(),
                previous_state=previous,
                state=state,
                detail=detail,
            )
        )

    def _abort(self, error: Exception) -> None:
        if self.state != WorkflowState.ABORTED:
            self._transition(WorkflowState.ABORTED, f"{type(error).__name__}: {error}")

    def _require_pending(self) -> MotionPlan:
        expected = {
            WorkflowState.AWAITING_GRASP_APPROVAL,
            WorkflowState.AWAITING_PLACE_APPROVAL,
        }
        if self.state not in expected or self._pending_plan is None:
            raise TransitionError(f"no plan can be reviewed while state={self.state.value}")
        return self._pending_plan

    def _validate_plan_identity(
        self,
        plan: MotionPlan,
        snapshot: SceneSnapshot,
        expected_phase: MotionPhase,
    ) -> None:
        assert self._task is not None
        if plan.task_id != self._task.task_id:
            raise SafetyViolation("planner returned a plan for a different task")
        if plan.scene_id != snapshot.scene_id:
            raise SafetyViolation("planner returned a plan for a different scene")
        if plan.start_state != snapshot.robot_state:
            raise SafetyViolation("planner changed the observed start state")
        self._safety.validate_reviewable(plan, expected_phase=expected_phase)

    @staticmethod
    def _validate_analysis(snapshot: SceneSnapshot, analysis: SceneAnalysis) -> None:
        if analysis.scene_id != snapshot.scene_id:
            raise ServiceError("perception returned an analysis for a different scene")
        if analysis.target.object_id == analysis.receptacle.object_id:
            raise ServiceError("target and receptacle must be different objects")

    def start(self, task: TaskRequest) -> PendingReview:
        if self.state != WorkflowState.IDLE:
            raise TransitionError(f"workflow cannot start while state={self.state.value}")
        self._task = task
        try:
            self._transition(WorkflowState.SENSING, "capture synchronized three-camera scene")
            snapshot = self._scene_provider.capture()
            self._safety.validate_snapshot(snapshot)
            self._snapshot = snapshot

            self._transition(WorkflowState.PERCEIVING, "localize target and receptacle")
            analysis = self._perception.analyze(snapshot, task)
            self._validate_analysis(snapshot, analysis)
            self._analysis = analysis

            self._transition(WorkflowState.GRASP_PLANNING, "rank collision-free grasps")
            candidates = tuple(self._grasp_service.propose(snapshot, analysis))
            valid = sorted(
                (
                    candidate
                    for candidate in candidates
                    if candidate.collision_free
                    and candidate.minimum_clearance_m >= self._safety.minimum_clearance_m
                ),
                key=lambda candidate: (candidate.score, candidate.minimum_clearance_m),
                reverse=True,
            )
            if not valid:
                raise ServiceError("no grasp candidate satisfies collision and clearance policy")
            self._selected_grasp = valid[0]
            plan = self._planner.plan_grasp(task, snapshot, analysis, valid)
            self._validate_plan_identity(plan, snapshot, MotionPhase.GRASP)
            if plan.arm != self._selected_grasp.arm:
                raise SafetyViolation("planner selected an arm inconsistent with the ranked grasp")
            if dict(plan.metadata).get("candidate_id") != self._selected_grasp.candidate_id:
                raise SafetyViolation("planner did not bind the plan to the selected grasp candidate")
            self._pending_plan = plan
            self._transition(
                WorkflowState.AWAITING_GRASP_APPROVAL,
                f"review grasp preview and digest {plan.digest}",
            )
            return PendingReview.from_plan(plan)
        except WorkflowError as error:
            self._abort(error)
            raise
        except Exception as error:
            wrapped = ServiceError(f"unexpected service failure: {type(error).__name__}: {error}")
            self._abort(wrapped)
            raise wrapped from error

    def approve_current(
        self,
        *,
        presented_digest: str,
        reviewer: str,
        now_s: float | None = None,
    ) -> ApprovalToken:
        plan = self._require_pending()
        return self._approvals.approve(
            plan,
            presented_digest=presented_digest,
            reviewer=reviewer,
            now_s=now_s,
        )

    def execute_current(
        self,
        token: ApprovalToken,
        *,
        now_s: float | None = None,
    ) -> PendingReview | WorkflowOutcome:
        plan = self._require_pending()
        assert self._task is not None
        assert self._snapshot is not None
        try:
            self._safety.validate_pre_execution(
                plan,
                current_state=self._executor.current_state(),
                scene_is_current=self._scene_provider.scene_is_current(plan.scene_id),
                now_s=now_s,
            )
            self._approvals.consume(plan, token, now_s=now_s)

            if plan.phase == MotionPhase.GRASP:
                self._transition(WorkflowState.EXECUTING_GRASP, "consume approval and execute grasp")
            else:
                self._transition(WorkflowState.EXECUTING_PLACE, "consume approval and execute place")
            execution = self._executor.execute(plan)
            if not execution.success:
                raise ServiceError(f"executor rejected plan: {execution.message}")

            if plan.phase == MotionPhase.GRASP:
                self._transition(WorkflowState.VERIFYING_GRASP, "verify held object")
                verification = self._verifier.verify_grasp(
                    self._task, self._snapshot, execution
                )
                if not verification.success:
                    raise ServiceError(f"grasp verification failed: {verification.message}")
                self._grasp_verification = verification
                return self._prepare_place()

            self._transition(WorkflowState.VERIFYING_PLACE, "verify object inside receptacle")
            verification = self._verifier.verify_place(
                self._task, self._snapshot, execution
            )
            if not verification.success:
                raise ServiceError(f"place verification failed: {verification.message}")
            if self._grasp_verification is None:
                raise TransitionError("place completed without a verified grasp")
            self._pending_plan = None
            self._transition(WorkflowState.DONE, "pick-and-place verified")
            return WorkflowOutcome(
                task_id=self._task.task_id,
                state=self.state,
                grasp_verification=self._grasp_verification,
                place_verification=verification,
            )
        except WorkflowError as error:
            self._abort(error)
            raise
        except Exception as error:
            wrapped = ServiceError(f"unexpected service failure: {type(error).__name__}: {error}")
            self._abort(wrapped)
            raise wrapped from error

    def _prepare_place(self) -> PendingReview:
        assert self._task is not None
        assert self._selected_grasp is not None
        self._transition(WorkflowState.PLACE_PLANNING, "recapture scene and plan with attached object")
        snapshot = self._scene_provider.capture()
        self._safety.validate_snapshot(snapshot)
        analysis = self._perception.analyze(snapshot, self._task)
        self._validate_analysis(snapshot, analysis)
        plan = self._planner.plan_place(
            self._task,
            snapshot,
            analysis,
            self._selected_grasp,
        )
        self._validate_plan_identity(plan, snapshot, MotionPhase.PLACE)
        if plan.arm != self._selected_grasp.arm:
            raise SafetyViolation("place arm differs from the verified grasp arm")
        if plan.collision_report.attached_object_id != analysis.target.object_id:
            raise SafetyViolation("place collision model is attached to the wrong object")
        self._snapshot = snapshot
        self._analysis = analysis
        self._pending_plan = plan
        self._transition(
            WorkflowState.AWAITING_PLACE_APPROVAL,
            f"review place preview and digest {plan.digest}",
        )
        return PendingReview.from_plan(plan)
