"""Iterative near-view framing policy for object observations.

This module is deliberately hardware-agnostic.  Callers provide two motion
solvers/executors and a fresh detector callback, which keeps the policy fully
testable without importing a robot SDK or touching a real arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union


class BoxEdge(str, Enum):
    LEFT = "left"
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")


@dataclass(frozen=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @classmethod
    def from_xyxy(cls, values: Sequence[float]) -> "BoundingBox":
        if len(values) != 4:
            raise ValueError("bounding box must contain four xyxy values")
        return cls(*(float(value) for value in values))

    def validate_for(self, image_size: ImageSize) -> None:
        if not (0 <= self.x_min < self.x_max <= image_size.width):
            raise ValueError("bounding box x coordinates are outside the image")
        if not (0 <= self.y_min < self.y_max <= image_size.height):
            raise ValueError("bounding box y coordinates are outside the image")


@dataclass(frozen=True)
class ObservationRefinementConfig:
    edge_margin_ratio: float = 0.10
    min_edge_margin_px: int = 0
    height_step_m: float = 0.03
    translation_step_m: float = 0.03
    rotation_step_rad: float = math.radians(8.0)
    max_iterations: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.edge_margin_ratio < 0.5:
            raise ValueError("edge_margin_ratio must be between 0 and 0.5")
        if self.min_edge_margin_px < 0:
            raise ValueError("min_edge_margin_px cannot be negative")
        if self.height_step_m <= 0 or self.translation_step_m <= 0:
            raise ValueError("translation steps must be positive")
        if self.rotation_step_rad <= 0:
            raise ValueError("rotation_step_rad must be positive")
        if not 1 <= self.max_iterations <= 5:
            raise ValueError("max_iterations must be between 1 and 5")


class AdjustmentKind(str, Enum):
    NONE = "none"
    HEIGHT = "height"
    TRANSLATE = "translate"
    ROTATE = "rotate"


@dataclass(frozen=True)
class CameraAdjustment:
    """A camera-local incremental motion.

    ``x_m`` is positive toward image-right, ``y_m`` toward image-bottom and
    ``z_m`` upward/away from the scene.  Angles are radians.  The hardware
    adapter is responsible for converting this camera-local delta through the
    current camera extrinsic transform before asking IK for a solution.
    """

    kind: AdjustmentKind
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    reason: str = ""
    fallback_for: Optional[AdjustmentKind] = None


@dataclass(frozen=True)
class AdjustmentDecision:
    near_edges: Tuple[BoxEdge, ...]
    primary: CameraAdjustment
    height_fallbacks: Tuple[CameraAdjustment, ...] = ()
    rotation_fallbacks: Tuple[CameraAdjustment, ...] = ()

    @property
    def good_view(self) -> bool:
        return self.primary.kind is AdjustmentKind.NONE


class RefinementStatus(str, Enum):
    GOOD = "good"
    MAX_ITERATIONS = "max_iterations"
    NO_SOLUTION = "no_solution"
    CAPTURE_FAILED = "capture_failed"


@dataclass(frozen=True)
class ObservationAdjustmentRecord:
    iteration: int
    box_before: BoundingBox
    near_edges: Tuple[BoxEdge, ...]
    primary: CameraAdjustment
    primary_solved: bool
    rotation_attempts: Tuple[CameraAdjustment, ...]
    applied: Optional[CameraAdjustment]
    box_after: Optional[BoundingBox] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ObservationRefinementResult:
    status: RefinementStatus
    final_box: BoundingBox
    history: Tuple[ObservationAdjustmentRecord, ...] = field(default_factory=tuple)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status is RefinementStatus.GOOD

    @property
    def iterations(self) -> int:
        return len(self.history)


def _near_edges(
    box: BoundingBox,
    image_size: ImageSize,
    config: ObservationRefinementConfig,
) -> Tuple[BoxEdge, ...]:
    box.validate_for(image_size)
    horizontal_margin = max(
        config.min_edge_margin_px,
        image_size.width * config.edge_margin_ratio,
    )
    vertical_margin = max(
        config.min_edge_margin_px,
        image_size.height * config.edge_margin_ratio,
    )
    edges = []
    if box.x_min <= horizontal_margin:
        edges.append(BoxEdge.LEFT)
    if box.y_min <= vertical_margin:
        edges.append(BoxEdge.TOP)
    if image_size.width - box.x_max <= horizontal_margin:
        edges.append(BoxEdge.RIGHT)
    if image_size.height - box.y_max <= vertical_margin:
        edges.append(BoxEdge.BOTTOM)
    return tuple(edges)


def _rotation_fallbacks(
    edges: Iterable[BoxEdge],
    config: ObservationRefinementConfig,
    primary_kind: AdjustmentKind,
) -> Tuple[CameraAdjustment, ...]:
    edge_set = set(edges)
    step = config.rotation_step_rad
    yaw = (-step if BoxEdge.LEFT in edge_set else 0.0) + (
        step if BoxEdge.RIGHT in edge_set else 0.0
    )
    pitch = (step if BoxEdge.TOP in edge_set else 0.0) + (
        -step if BoxEdge.BOTTOM in edge_set else 0.0
    )

    candidates = []
    if yaw or pitch:
        candidates.append((0.0, pitch, yaw, "rotate camera toward offending edge(s)"))
    if yaw and pitch:
        candidates.extend(
            [
                (0.0, 0.0, yaw, "yaw-only fallback"),
                (0.0, pitch, 0.0, "pitch-only fallback"),
            ]
        )
    if not candidates:
        # When the object spans opposite edges, rotation cannot reduce its
        # scale, but a small alternate view may still restore a usable framing
        # when the height/retreat motion is outside the arm workspace.
        candidates.extend(
            [
                (step, 0.0, 0.0, "roll fallback for blocked height motion"),
                (0.0, step, 0.0, "pitch fallback for blocked height motion"),
                (0.0, 0.0, step, "yaw fallback for blocked height motion"),
            ]
        )

    return tuple(
        CameraAdjustment(
            kind=AdjustmentKind.ROTATE,
            roll_rad=roll,
            pitch_rad=pitch_value,
            yaw_rad=yaw_value,
            reason=reason,
            fallback_for=primary_kind,
        )
        for roll, pitch_value, yaw_value, reason in candidates
    )


def plan_observation_adjustment(
    box: BoundingBox,
    image_size: ImageSize,
    config: ObservationRefinementConfig = ObservationRefinementConfig(),
) -> AdjustmentDecision:
    """Return the next camera-local adjustment for a detected object box."""

    edges = _near_edges(box, image_size, config)
    edge_set = set(edges)
    if not edges:
        return AdjustmentDecision(
            near_edges=(),
            primary=CameraAdjustment(
                kind=AdjustmentKind.NONE,
                reason="box has sufficient margin on all four sides",
            ),
        )

    horizontal_margin = max(
        config.min_edge_margin_px,
        image_size.width * config.edge_margin_ratio,
    )
    vertical_margin = max(
        config.min_edge_margin_px,
        image_size.height * config.edge_margin_ratio,
    )
    cannot_fit_safe_interior = (
        box.x_max - box.x_min >= image_size.width - 2.0 * horizontal_margin
        or box.y_max - box.y_min >= image_size.height - 2.0 * vertical_margin
    )
    opposite_pair = (
        {BoxEdge.TOP, BoxEdge.BOTTOM}.issubset(edge_set)
        or {BoxEdge.LEFT, BoxEdge.RIGHT}.issubset(edge_set)
    )
    if len(edges) >= 3 or opposite_pair or cannot_fit_safe_interior:
        primary = CameraAdjustment(
            kind=AdjustmentKind.HEIGHT,
            z_m=config.height_step_m,
            reason=(
                "box is too large for the safe image interior"
                if cannot_fit_safe_interior
                else "box spans opposite or at least three image edges"
            ),
        )
    else:
        x_m = (-config.translation_step_m if BoxEdge.LEFT in edge_set else 0.0) + (
            config.translation_step_m if BoxEdge.RIGHT in edge_set else 0.0
        )
        y_m = (-config.translation_step_m if BoxEdge.TOP in edge_set else 0.0) + (
            config.translation_step_m if BoxEdge.BOTTOM in edge_set else 0.0
        )
        primary = CameraAdjustment(
            kind=AdjustmentKind.TRANSLATE,
            x_m=x_m,
            y_m=y_m,
            reason="move camera toward offending image edge(s)",
        )

    return AdjustmentDecision(
        near_edges=edges,
        primary=primary,
        height_fallbacks=(
            (
                CameraAdjustment(
                    kind=AdjustmentKind.HEIGHT,
                    z_m=config.height_step_m,
                    reason="translation unavailable: raise camera by 3 cm",
                    fallback_for=AdjustmentKind.TRANSLATE,
                ),
            )
            if primary.kind is AdjustmentKind.TRANSLATE
            else ()
        ),
        rotation_fallbacks=_rotation_fallbacks(edges, config, primary.kind),
    )


Solver = Callable[[CameraAdjustment], object]
BoxCapture = Callable[[], Union[BoundingBox, Sequence[float]]]


def _solved(value: object) -> bool:
    # Several robot SDKs return integer 0 for success.  Only explicit False or
    # None means that the requested motion has no solution.
    return value is not None and value is not False


class ObservationRefiner:
    """Run detect-adjust-detect for at most five camera adjustments."""

    def __init__(
        self,
        image_size: ImageSize,
        solve_translation: Solver,
        solve_rotation: Solver,
        capture_box: BoxCapture,
        config: ObservationRefinementConfig = ObservationRefinementConfig(),
    ) -> None:
        self._image_size = image_size
        self._solve_translation = solve_translation
        self._solve_rotation = solve_rotation
        self._capture_box = capture_box
        self._config = config

    def _capture(self) -> BoundingBox:
        captured = self._capture_box()
        if isinstance(captured, BoundingBox):
            box = captured
        else:
            box = BoundingBox.from_xyxy(captured)
        box.validate_for(self._image_size)
        return box

    def run(self, initial_box: Union[BoundingBox, Sequence[float]]) -> ObservationRefinementResult:
        box = (
            initial_box
            if isinstance(initial_box, BoundingBox)
            else BoundingBox.from_xyxy(initial_box)
        )
        box.validate_for(self._image_size)
        history = []

        for iteration in range(1, self._config.max_iterations + 1):
            decision = plan_observation_adjustment(box, self._image_size, self._config)
            if decision.good_view:
                return ObservationRefinementResult(
                    status=RefinementStatus.GOOD,
                    final_box=box,
                    history=tuple(history),
                )

            primary_solved = False
            primary_error = None
            try:
                primary_solved = _solved(self._solve_translation(decision.primary))
            except Exception as exc:  # adapter/IK errors trigger rotation fallback
                primary_error = f"{type(exc).__name__}: {exc}"

            applied = decision.primary if primary_solved else None
            attempted_rotations = []
            rotation_errors = []
            if not primary_solved:
                for fallback in decision.height_fallbacks:
                    try:
                        if _solved(self._solve_translation(fallback)):
                            applied = fallback
                            break
                    except Exception as exc:
                        rotation_errors.append(f"{type(exc).__name__}: {exc}")

            if applied is None:
                for fallback in decision.rotation_fallbacks:
                    attempted_rotations.append(fallback)
                    try:
                        if _solved(self._solve_rotation(fallback)):
                            applied = fallback
                            break
                    except Exception as exc:
                        rotation_errors.append(f"{type(exc).__name__}: {exc}")

            if applied is None:
                error_parts = [part for part in [primary_error, *rotation_errors] if part]
                error = "; ".join(error_parts) or "no IK solution for translation, height, or rotation"
                history.append(
                    ObservationAdjustmentRecord(
                        iteration=iteration,
                        box_before=box,
                        near_edges=decision.near_edges,
                        primary=decision.primary,
                        primary_solved=False,
                        rotation_attempts=tuple(attempted_rotations),
                        applied=None,
                        error=error,
                    )
                )
                return ObservationRefinementResult(
                    status=RefinementStatus.NO_SOLUTION,
                    final_box=box,
                    history=tuple(history),
                    error=error,
                )

            try:
                box_after = self._capture()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                history.append(
                    ObservationAdjustmentRecord(
                        iteration=iteration,
                        box_before=box,
                        near_edges=decision.near_edges,
                        primary=decision.primary,
                        primary_solved=primary_solved,
                        rotation_attempts=tuple(attempted_rotations),
                        applied=applied,
                        error=error,
                    )
                )
                return ObservationRefinementResult(
                    status=RefinementStatus.CAPTURE_FAILED,
                    final_box=box,
                    history=tuple(history),
                    error=error,
                )

            history.append(
                ObservationAdjustmentRecord(
                    iteration=iteration,
                    box_before=box,
                    near_edges=decision.near_edges,
                    primary=decision.primary,
                    primary_solved=primary_solved,
                    rotation_attempts=tuple(attempted_rotations),
                    applied=applied,
                    box_after=box_after,
                )
            )
            box = box_after

        final_decision = plan_observation_adjustment(box, self._image_size, self._config)
        status = RefinementStatus.GOOD if final_decision.good_view else RefinementStatus.MAX_ITERATIONS
        return ObservationRefinementResult(
            status=status,
            final_box=box,
            history=tuple(history),
            error=None if status is RefinementStatus.GOOD else "box still touches an edge margin",
        )
