from __future__ import annotations

import unittest
from pathlib import Path

from robot_workflow.gripper_clearance import (
    KINEMATIC_TCP_LENGTH_M,
    MINIMUM_TABLE_CLEARANCE_M,
    SIMULATED_OPEN_MESH_LENGTH_M,
    VIRTUAL_COLLISION_LENGTH_M,
    approach_world_z,
    candidate_advances_m,
    virtual_tip_clearance_m,
)


class GripperClearanceTests(unittest.TestCase):
    def test_virtual_envelope_corrects_mesh_shortfall_without_moving_tcp(self) -> None:
        self.assertAlmostEqual(KINEMATIC_TCP_LENGTH_M, 0.172)
        self.assertAlmostEqual(SIMULATED_OPEN_MESH_LENGTH_M, 0.156192)
        self.assertAlmostEqual(VIRTUAL_COLLISION_LENGTH_M, 0.1755)
        self.assertAlmostEqual(MINIMUM_TABLE_CLEARANCE_M, 0.015)
        self.assertAlmostEqual(
            VIRTUAL_COLLISION_LENGTH_M - SIMULATED_OPEN_MESH_LENGTH_M,
            0.019308,
        )
        self.assertAlmostEqual(
            VIRTUAL_COLLISION_LENGTH_M - KINEMATIC_TCP_LENGTH_M,
            0.0035,
        )

    def test_advance_trials_include_zero_and_are_deepest_first(self) -> None:
        advances = candidate_advances_m(0.03)
        self.assertAlmostEqual(advances[0], 0.035)
        self.assertEqual(
            [round(value, 3) for value in advances[1:]],
            [0.03, 0.025, 0.02, 0.015, 0.01, 0.005, 0.0],
        )
        self.assertEqual(candidate_advances_m(0.005), (0.01, 0.005, 0.0))

    def test_incident_candidate_rejects_full_depth_but_accepts_five_mm(self) -> None:
        pose = [
            0.37873490664386744,
            -0.025270776294703225,
            0.025745309769037883,
            -3.14151028723629,
            1.0552359282571573,
            -3.095345094759489,
        ]
        approach_z = approach_world_z(pose)

        def advanced(depth_m: float) -> list[float]:
            result = list(pose)
            result[2] += depth_m * approach_z
            return result

        full_depth_clearance = virtual_tip_clearance_m(advanced(0.03))
        five_mm_clearance = virtual_tip_clearance_m(advanced(0.005))
        self.assertLess(full_depth_clearance, MINIMUM_TABLE_CLEARANCE_M)
        self.assertGreaterEqual(five_mm_clearance, MINIMUM_TABLE_CLEARANCE_M)
        self.assertAlmostEqual(full_depth_clearance, -0.003400, places=5)
        self.assertAlmostEqual(five_mm_clearance, 0.018350, places=5)

    def test_virtual_length_cannot_be_shorter_than_tcp(self) -> None:
        with self.assertRaises(ValueError):
            virtual_tip_clearance_m(
                [0, 0, 0.1, 0, 0, 0],
                virtual_length_m=0.16,
                tcp_length_m=0.172,
            )

    def test_hardware_path_is_the_exact_simulation_path(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_pistol_grasp_workflow.py"
        ).read_text("utf-8")
        self.assertIn('candidate_path = candidate["simulation_waypoints"]', source)
        self.assertIn("path = candidate_path", source)
        self.assertNotIn('path = selected["legacy_waypoints"]', source)
        self.assertIn("hardware path is not exactly the simulated path", source)
        self.assertIn("simulation_path_sha256", source)

    def test_graspnet_pose_is_only_a_reference_between_retracted_prepose_and_final(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "evaluate_pistol_barrel_grasps.py"
        ).read_text("utf-8")
        self.assertIn("PREGRASP_RETRACTION_M = 0.05", source)
        self.assertIn(
            "pregrasp_pose = kinematics.cal_pre_pose(\n"
            "                grasp_pose, PREGRASP_RETRACTION_M",
            source,
        )
        self.assertIn("pre_end = kinematics.gripper_to_end(pregrasp_pose)", source)
        self.assertNotIn("pre_end = kinematics.gripper_to_end(grasp_pose)", source)
        self.assertIn(
            '"approach_distance_m": PREGRASP_RETRACTION_M + advance_m',
            source,
        )
        workflow_source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_pistol_grasp_workflow.py"
        ).read_text("utf-8")
        self.assertIn("REQUIRED_PREGRASP_RETRACTION_M = 0.05", workflow_source)
        self.assertIn("MAX_APPROACH_LATERAL_DEVIATION_M = 0.005", workflow_source)
        self.assertIn(
            '"pregrasp-to-final path is not sufficiently straight"',
            workflow_source,
        )
        self.assertIn(
            '"pregrasp-to-final path does not advance monotonically"',
            workflow_source,
        )

    def test_closed_gripper_lifts_then_places_in_green_box(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_pistol_grasp_workflow.py"
        ).read_text("utf-8")
        self.assertIn("POST_GRASP_LIFT_M = 0.10", source)
        self.assertIn("POST_GRASP_LIFT_SEGMENTS = 4", source)
        self.assertIn(
            "POST_GRASP_LIFT_M * segment / POST_GRASP_LIFT_SEGMENTS",
            source,
        )
        self.assertIn('hardware_video.mark("post_grasp_vertical_lift")', source)
        self.assertIn('state["lift_completed"] = True', source)
        self.assertIn("post-grasp lift is not sufficiently vertical", source)
        self.assertIn("post-grasp lift does not rise monotonically", source)
        self.assertIn(
            'state["status"] = "placed_in_green_box_and_retreated_100mm"',
            source,
        )
        self.assertIn("BOX_DROP_X_M = 0.45", source)
        self.assertIn("BOX_DROP_Y_M = -0.21", source)
        self.assertIn("BOX_RIM_Z_M = 0.058", source)
        self.assertIn("TRANSFER_TRAVEL_Z_M = 0.300", source)
        self.assertIn('extra_lift_ceiling = None', source)
        self.assertIn('"highest_reachable_z_m": float(pose[2])', source)
        self.assertIn('"minimum_safe_transfer_z_m": minimum_safe_transfer_z_m', source)
        self.assertIn("TRANSFER_TRANSITION_Y_M = 0.05", source)
        self.assertIn(
            "TRANSFER_REORIENT_RPY_RAD = (math.pi, 1.0, -math.pi / 4.0)",
            source,
        )
        self.assertIn("TRANSFER_REORIENT_SEGMENTS = 1", source)
        self.assertIn('"horizontal_to_transition"', source)
        self.assertIn('f"stationary_reorient_{index}"', source)
        self.assertIn("TRANSFER_RELEASE_GRIPPER_Z_M = 0.185", source)
        self.assertIn("TRANSFER_RETREAT_M = 0.10", source)
        self.assertIn('hardware_video.mark("green_box_release")', source)
        self.assertIn('state["box_release_sent"] = True', source)
        self.assertIn('state["box_retreat_completed"] = True', source)
        self.assertNotIn("held object would not clear the green-box rim", source)
        self.assertIn('"held_object_box_rim_contact_allowed": True', source)
        self.assertIn('"predicted_object_rim_overlap_m": predicted_object_rim_overlap_m', source)
        self.assertIn("green-box release height is outside the configured drop window", source)
        self.assertIn("gateway.simulate_sequence(", source)
        self.assertIn('"box_transfer_full_sequence.mp4"', source)
        self.assertNotIn("box_transfer_{len(segments)", source)
        self.assertIn('state["post_lift_hold_validation"]', source)
        self.assertIn('state.setdefault("transfer_hold_validations", [])', source)
        self.assertIn("object retention check failed after initial lift", source)
        self.assertIn('state["post_lift_visual_retention"]', source)
        self.assertIn('state.setdefault("transfer_visual_retentions", [])', source)
        self.assertIn("wrist-camera retention check failed after initial lift", source)
        self.assertIn("MAXIMUM_WRIST_NEAR_DEPTH_INCREASE_MM = 40.0", source)
        self.assertNotIn('hardware_video.mark("reverse_return_motion")', source)

    def test_real_hardware_video_covers_grasp_and_return(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_pistol_grasp_workflow.py"
        ).read_text(encoding="utf-8")
        self.assertIn("HardwareVideoRecorder", source)
        self.assertLess(
            source.index('hardware_video.mark("gripper_opening")'),
            source.index("rm_set_gripper_release"),
        )
        self.assertLess(
            source.index('hardware_video.mark("gripper_closing")'),
            source.index("rm_set_gripper_pick"),
        )
        self.assertIn('hardware_video.mark("post_grasp_vertical_lift")', source)
        self.assertIn('hardware_video.mark("green_box_release")', source)
        self.assertIn('state["hardware_video"] = hardware_video.stop()', source)
        self.assertIn('state["lift_completed"] = True', source)
        self.assertIn('state["lift_sent"] = True', source)
        self.assertIn('state["box_transfer_sent"] = True', source)
        self.assertIn('state["box_release_sent"] = True', source)
        self.assertNotIn("joints_list[::-1]", source)

    def test_relaxed_quality_mode_keeps_hard_safety_gates(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_pistol_grasp_workflow.py"
        ).read_text("utf-8")
        self.assertIn("--relaxed-grasp-quality", source)
        self.assertIn("RELAXED_CROSS_SECTION_WIDTH_M = (0.012, 0.060)", source)
        self.assertIn("RELAXED_MINIMUM_SIDE_CLEARANCE_M = -0.005", source)
        self.assertIn("PREFERRED_MINIMUM_SIDE_CLEARANCE_M = 0.0", source)
        self.assertIn('result["preferred_physical_grasp"]', source)
        self.assertIn("(1.0 if preferred_physical_grasp else 0.0)", source)
        self.assertIn("MINIMUM_JOINT_MARGIN_DEG = 3.0", source)
        self.assertIn("margins.min()) < MINIMUM_JOINT_MARGIN_DEG", source)
        self.assertIn("MAX_WRIST7_TO_PREGRASP_DELTA_DEG = 90.0", source)
        self.assertIn("wrist-7 rotation to pregrasp exceeds limit", source)
        self.assertIn("final virtual tip does not meet table clearance", source)
        self.assertIn("hardware path is not exactly the simulated path", source)

    def test_fast_near_view_uses_verified_height_without_recapture_loop(self) -> None:
        root = Path(__file__).resolve().parents[1]
        planner = (root / "scripts" / "plan_mask_centered_observation.py").read_text(
            encoding="utf-8"
        )
        executor = (root / "scripts" / "execute_observation_plan.py").read_text(
            encoding="utf-8"
        )
        workflow = (root / "scripts" / "run_pistol_grasp_workflow.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("FAST_NEAR_VIEW_CAMERA_Z_M = 0.354", planner)
        self.assertIn("FAST_NEAR_VIEW_STEP_M = 0.020", planner)
        self.assertIn('recording=False', planner)
        self.assertIn('selected_attempt["waypoints"] = combined_waypoints', planner)
        self.assertIn("MAXIMUM_OBSERVATION_WAYPOINTS = 75", executor)
        self.assertIn('parser.add_argument("--max-refinements", type=int, default=0)', workflow)
        self.assertIn("0 <= arguments.max_refinements <= 8", workflow)

    def test_magazine_workflow_uses_the_whole_object_mask(self) -> None:
        root = Path(__file__).resolve().parents[1]
        evaluator = (root / "scripts" / "evaluate_pistol_barrel_grasps.py").read_text(
            encoding="utf-8"
        )
        workflow = (root / "scripts" / "run_pistol_grasp_workflow.py").read_text(
            encoding="utf-8"
        )
        magazine = (root / "scripts" / "run_magazine_grasp_workflow.py").read_text(
            encoding="utf-8"
        )
        capture = (root / "scripts" / "capture_dual_wrist_target.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--whole-object",', evaluator)
        self.assertIn('"grasp_region": "whole_object"', evaluator)
        self.assertIn('parser.add_argument("--target-text", default="pistol")', workflow)
        self.assertIn('parser.add_argument("--whole-object-grasp"', workflow)
        self.assertIn('"rectangular black object"', magazine)
        self.assertIn('"--whole-object-grasp"', magazine)
        self.assertIn('default=0.25', capture)
        self.assertIn('"reason": "mask_too_large"', capture)
        self.assertIn('.casefold() != right["label"].casefold()', capture)
        self.assertIn('"analytic_topdown_prism"', evaluator)
        self.assertIn("regular_prism_mode=arguments.whole_object_grasp", workflow)
        self.assertIn('"magazine_centered_topdown_clearance_and_joint_margin"', workflow)
        self.assertIn("MAGAZINE_MAX_WRIST7_TO_PREGRASP_DELTA_DEG = 140.0", workflow)
        self.assertIn('"two_stage_stationary_reorient_for_magazine"', workflow)
        self.assertIn("maximum_attempts = 4", workflow)
        self.assertIn("regular_prism_mode=arguments.whole_object_grasp", workflow)

    def test_full_workflow_releases_left_gripper_before_initial_reset(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / "scripts" / "run_pistol_grasp_workflow.py").read_text(
            "utf-8"
        )
        initial = (root / "scripts" / "move_dual_wrist_observation.py").read_text(
            "utf-8"
        )
        self.assertIn('"--release-left-before-initial"', workflow)
        self.assertIn('parser.add_argument("--release-left-before-initial"', initial)
        self.assertLess(
            initial.index("rm_set_gripper_release"),
            initial.index("target = target_joints(arguments.stage, arm, center)"),
        )
        self.assertIn('"completed_before_arm_motion": True', initial)


if __name__ == "__main__":
    unittest.main()
