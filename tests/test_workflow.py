from __future__ import annotations

from dataclasses import replace
import unittest

from robot_workflow.approval import ApprovalToken
from robot_workflow.domain import GraspCandidate, TaskRequest, WorkflowState
from robot_workflow.errors import SafetyViolation, ServiceError, TransitionError
from robot_workflow.offline import (
    OfflineGraspService,
    OfflineMotionPlanner,
    OfflinePerception,
    OfflineVerifier,
    SimulatedExecutor,
    SimulationWorld,
    SyntheticThreeCameraSceneProvider,
)
from robot_workflow.protocols import ExecutorMode
from robot_workflow.workflow import PickPlaceWorkflow, WorkflowOutcome


def build_workflow() -> tuple[PickPlaceWorkflow, SimulationWorld, SimulatedExecutor, OfflineVerifier]:
    world = SimulationWorld()
    executor = SimulatedExecutor(world)
    verifier = OfflineVerifier(world)
    workflow = PickPlaceWorkflow(
        scene_provider=SyntheticThreeCameraSceneProvider(world),
        perception=OfflinePerception(),
        grasp_service=OfflineGraspService(),
        planner=OfflineMotionPlanner(),
        executor=executor,
        verifier=verifier,
    )
    return workflow, world, executor, verifier


class WorkflowHappyPathTests(unittest.TestCase):
    def test_two_distinct_human_reviews_complete_simulated_task(self) -> None:
        workflow, world, executor, _ = build_workflow()
        grasp_review = workflow.start(TaskRequest("red block", "blue box", task_id="task-e2e"))
        self.assertEqual(workflow.state, WorkflowState.AWAITING_GRASP_APPROVAL)
        self.assertEqual(grasp_review.phase.value, "grasp")

        grasp_token = workflow.approve_current(
            presented_digest=grasp_review.plan_digest,
            reviewer="offline-tester",
        )
        place_review = workflow.execute_current(grasp_token)
        self.assertEqual(workflow.state, WorkflowState.AWAITING_PLACE_APPROVAL)
        self.assertEqual(place_review.phase.value, "place")
        self.assertIsNotNone(world.held_object_id)

        place_token = workflow.approve_current(
            presented_digest=place_review.plan_digest,
            reviewer="offline-tester",
        )
        outcome = workflow.execute_current(place_token)
        self.assertIsInstance(outcome, WorkflowOutcome)
        self.assertEqual(outcome.state, WorkflowState.DONE)
        self.assertEqual(len(executor.history), 2)
        self.assertIsNone(world.held_object_id)
        self.assertEqual(world.placed_object_id, "target-1")
        self.assertEqual(
            [event.state for event in workflow.audit_log],
            [
                WorkflowState.SENSING,
                WorkflowState.PERCEIVING,
                WorkflowState.GRASP_PLANNING,
                WorkflowState.AWAITING_GRASP_APPROVAL,
                WorkflowState.EXECUTING_GRASP,
                WorkflowState.VERIFYING_GRASP,
                WorkflowState.PLACE_PLANNING,
                WorkflowState.AWAITING_PLACE_APPROVAL,
                WorkflowState.EXECUTING_PLACE,
                WorkflowState.VERIFYING_PLACE,
                WorkflowState.DONE,
            ],
        )

    def test_execution_without_pending_review_is_impossible(self) -> None:
        workflow, _, _, _ = build_workflow()
        with self.assertRaises(TransitionError):
            workflow.approve_current(presented_digest="0" * 64, reviewer="operator")
        with self.assertRaises(TransitionError):
            workflow.execute_current(ApprovalToken("invented", "invented"))


class WorkflowFailureTests(unittest.TestCase):
    def test_scene_change_after_approval_aborts_without_motion(self) -> None:
        workflow, world, executor, _ = build_workflow()
        review = workflow.start(TaskRequest("object", "box"))
        token = workflow.approve_current(
            presented_digest=review.plan_digest,
            reviewer="offline-tester",
        )
        world.mutate()
        with self.assertRaisesRegex(SafetyViolation, "scene changed"):
            workflow.execute_current(token)
        self.assertEqual(workflow.state, WorkflowState.ABORTED)
        self.assertEqual(executor.history, [])

    def test_failed_grasp_verification_aborts_before_place_planning(self) -> None:
        workflow, _, executor, verifier = build_workflow()
        verifier.fail_grasp = True
        review = workflow.start(TaskRequest("object", "box"))
        token = workflow.approve_current(
            presented_digest=review.plan_digest,
            reviewer="offline-tester",
        )
        with self.assertRaisesRegex(ServiceError, "verification failed"):
            workflow.execute_current(token)
        self.assertEqual(workflow.state, WorkflowState.ABORTED)
        self.assertEqual(len(executor.history), 1)

    def test_no_safe_grasp_aborts_before_planner_or_executor(self) -> None:
        world = SimulationWorld()
        unsafe = GraspCandidate(
            "unsafe",
            "left",
            (0.0,) * 6,
            0.99,
            0.04,
            0.02,
            False,
            0.0,
        )
        executor = SimulatedExecutor(world)
        workflow = PickPlaceWorkflow(
            scene_provider=SyntheticThreeCameraSceneProvider(world),
            perception=OfflinePerception(),
            grasp_service=OfflineGraspService((unsafe,)),
            planner=OfflineMotionPlanner(),
            executor=executor,
            verifier=OfflineVerifier(world),
        )
        with self.assertRaisesRegex(ServiceError, "no grasp candidate"):
            workflow.start(TaskRequest("object", "box"))
        self.assertEqual(workflow.state, WorkflowState.ABORTED)
        self.assertEqual(executor.history, [])

    def test_unexpected_dependency_exception_is_wrapped_and_aborts(self) -> None:
        class BrokenPerception(OfflinePerception):
            def analyze(self, snapshot, task):
                raise KeyError("malformed model response")

        world = SimulationWorld()
        workflow = PickPlaceWorkflow(
            scene_provider=SyntheticThreeCameraSceneProvider(world),
            perception=BrokenPerception(),
            grasp_service=OfflineGraspService(),
            planner=OfflineMotionPlanner(),
            executor=SimulatedExecutor(world),
            verifier=OfflineVerifier(world),
        )
        with self.assertRaisesRegex(ServiceError, "unexpected service failure.*KeyError"):
            workflow.start(TaskRequest("object", "box"))
        self.assertEqual(workflow.state, WorkflowState.ABORTED)

    def test_hardware_executor_is_rejected_by_default(self) -> None:
        class HardwareExecutor(SimulatedExecutor):
            mode = ExecutorMode.HARDWARE

        world = SimulationWorld()
        with self.assertRaisesRegex(SafetyViolation, "hardware executor rejected"):
            PickPlaceWorkflow(
                scene_provider=SyntheticThreeCameraSceneProvider(world),
                perception=OfflinePerception(),
                grasp_service=OfflineGraspService(),
                planner=OfflineMotionPlanner(),
                executor=HardwareExecutor(world),
                verifier=OfflineVerifier(world),
            )


if __name__ == "__main__":
    unittest.main()
