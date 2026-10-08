"""Capture required wrist RGB-D while treating the fixed head camera as optional."""

from __future__ import annotations

from typing import Any


def capture_required_wrists_optional_head(
    environment: Any,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    observation: dict[str, dict[str, Any]] = {}
    for name, arm in (
        ("left", environment.arm_left),
        ("right", environment.arm_right),
    ):
        status, joints = arm.get_joint_degree()
        if status != 0 or len(joints) != 7:
            raise RuntimeError(f"failed to read required {name}-arm joints")
        arm_data: dict[str, Any] = {"joints": joints}
        arm_data.update(arm.camera.get_framedata())
        observation[name] = arm_data

    head_error = None
    try:
        observation["head"] = environment.head_camera.get_framedata()
    except RuntimeError as error:
        head_error = repr(error)
    return observation, head_error
