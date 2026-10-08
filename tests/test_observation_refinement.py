import unittest

from robot_workflow.observation_refinement import (
    AdjustmentKind,
    BoundingBox,
    BoxEdge,
    ImageSize,
    ObservationRefinementConfig,
    ObservationRefiner,
    RefinementStatus,
    plan_observation_adjustment,
)


SIZE = ImageSize(640, 480)


class ObservationPlanningTests(unittest.TestCase):
    def test_centered_box_needs_no_motion(self):
        decision = plan_observation_adjustment(BoundingBox(100, 100, 500, 380), SIZE)
        self.assertTrue(decision.good_view)
        self.assertEqual(decision.primary.kind, AdjustmentKind.NONE)

    def test_each_single_edge_translates_in_camera_image_direction(self):
        cases = [
            (BoundingBox(10, 100, 300, 300), BoxEdge.LEFT, (-0.03, 0.0)),
            (BoundingBox(100, 10, 500, 300), BoxEdge.TOP, (0.0, -0.03)),
            (BoundingBox(300, 100, 630, 300), BoxEdge.RIGHT, (0.03, 0.0)),
            (BoundingBox(100, 200, 500, 470), BoxEdge.BOTTOM, (0.0, 0.03)),
        ]
        for box, edge, expected in cases:
            with self.subTest(edge=edge):
                decision = plan_observation_adjustment(box, SIZE)
                self.assertEqual(decision.near_edges, (edge,))
                self.assertEqual(decision.primary.kind, AdjustmentKind.TRANSLATE)
                self.assertEqual((decision.primary.x_m, decision.primary.y_m), expected)

    def test_adjacent_edges_translate_diagonally(self):
        decision = plan_observation_adjustment(BoundingBox(10, 10, 400, 300), SIZE)
        self.assertEqual(decision.near_edges, (BoxEdge.LEFT, BoxEdge.TOP))
        self.assertEqual(decision.primary.kind, AdjustmentKind.TRANSLATE)
        self.assertEqual((decision.primary.x_m, decision.primary.y_m), (-0.03, -0.03))

    def test_translation_falls_back_to_three_centimeter_height(self):
        decision = plan_observation_adjustment(BoundingBox(10, 100, 400, 300), SIZE)

        self.assertEqual(decision.primary.kind, AdjustmentKind.TRANSLATE)
        self.assertEqual(len(decision.height_fallbacks), 1)
        self.assertEqual(decision.height_fallbacks[0].kind, AdjustmentKind.HEIGHT)
        self.assertEqual(decision.height_fallbacks[0].z_m, 0.03)

    def test_opposite_and_three_or_four_edges_raise_camera(self):
        boxes = [
            BoundingBox(100, 10, 500, 470),
            BoundingBox(10, 100, 630, 380),
            BoundingBox(10, 10, 630, 300),
            BoundingBox(10, 10, 630, 470),
        ]
        for box in boxes:
            with self.subTest(box=box):
                decision = plan_observation_adjustment(box, SIZE)
                self.assertEqual(decision.primary.kind, AdjustmentKind.HEIGHT)
                self.assertEqual(decision.primary.z_m, 0.03)

    def test_single_edge_but_oversized_box_raises_instead_of_oscillating(self):
        decision = plan_observation_adjustment(
            BoundingBox(215, 61, 515, 480), SIZE
        )

        self.assertEqual(decision.near_edges, (BoxEdge.BOTTOM,))
        self.assertEqual(decision.primary.kind, AdjustmentKind.HEIGHT)
        self.assertEqual(decision.primary.z_m, 0.03)
        self.assertIn("too large", decision.primary.reason)

    def test_iteration_limit_cannot_exceed_five(self):
        with self.assertRaises(ValueError):
            ObservationRefinementConfig(max_iterations=6)


class ObservationRefinerTests(unittest.TestCase):
    def test_failed_translation_uses_height_then_recaptures(self):
        translations = []
        rotations = []

        refiner = ObservationRefiner(
            SIZE,
            solve_translation=lambda motion: translations.append(motion) or (
                0 if motion.kind is AdjustmentKind.HEIGHT else None
            ),
            solve_rotation=lambda motion: rotations.append(motion) or 0,
            capture_box=lambda: BoundingBox(100, 100, 500, 380),
        )
        result = refiner.run(BoundingBox(10, 100, 400, 300))

        self.assertEqual(result.status, RefinementStatus.GOOD)
        self.assertEqual(len(translations), 2)
        self.assertEqual(len(rotations), 0)
        self.assertEqual(result.history[0].applied.kind, AdjustmentKind.HEIGHT)
        self.assertFalse(result.history[0].primary_solved)

    def test_no_translation_or_rotation_solution_stops_without_recapture(self):
        captures = []
        refiner = ObservationRefiner(
            SIZE,
            solve_translation=lambda _motion: None,
            solve_rotation=lambda _motion: None,
            capture_box=lambda: captures.append(True),
        )
        result = refiner.run(BoundingBox(10, 100, 400, 300))

        self.assertEqual(result.status, RefinementStatus.NO_SOLUTION)
        self.assertEqual(captures, [])
        self.assertGreaterEqual(len(result.history[0].rotation_attempts), 1)

    def test_persistent_bad_box_stops_after_exactly_five_adjustments(self):
        capture_count = 0

        def capture():
            nonlocal capture_count
            capture_count += 1
            return BoundingBox(10, 100, 400, 300)

        refiner = ObservationRefiner(
            SIZE,
            solve_translation=lambda _motion: True,
            solve_rotation=lambda _motion: True,
            capture_box=capture,
        )
        result = refiner.run(BoundingBox(10, 100, 400, 300))

        self.assertEqual(result.status, RefinementStatus.MAX_ITERATIONS)
        self.assertEqual(result.iterations, 5)
        self.assertEqual(capture_count, 5)

    def test_capture_exception_is_reported(self):
        def failing_capture():
            raise RuntimeError("camera unavailable")

        refiner = ObservationRefiner(
            SIZE,
            solve_translation=lambda _motion: True,
            solve_rotation=lambda _motion: True,
            capture_box=failing_capture,
        )
        result = refiner.run(BoundingBox(10, 100, 400, 300))

        self.assertEqual(result.status, RefinementStatus.CAPTURE_FAILED)
        self.assertIn("camera unavailable", result.error)


if __name__ == "__main__":
    unittest.main()
