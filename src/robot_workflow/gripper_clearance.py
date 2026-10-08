"""Conservative gripper-floor clearance calculations.

The legacy kinematic transform places the grasp TCP 172 mm from Link7, while
the open 4C2 collision mesh in the deployed simulator reaches only 156.192 mm.
The virtual collision tip deliberately extends 3.5 mm beyond the kinematic TCP.
This remains below the previous 5 mm extension while restoring 1 mm of margin
relative to the 2.5 mm midpoint trial.
It is used only for safety rejection; it never changes the commanded grasp TCP.
"""

from __future__ import annotations

import math
from typing import Sequence


KINEMATIC_TCP_LENGTH_M = 0.172
SIMULATED_OPEN_MESH_LENGTH_M = 0.156192
VIRTUAL_COLLISION_LENGTH_M = 0.1755
# Hardware trials at 7 mm repeatedly left the gripper near fully open after the
# close command, consistent with a finger contacting the table before enclosing
# the prop.  The last verified hold retained 16.17 mm, so use a 15 mm hard floor.
MINIMUM_TABLE_CLEARANCE_M = 0.015
TABLE_PLANE_Z_M = 0.0
FINAL_INSERTION_BIAS_M = 0.005


def approach_world_z(gripper_pose: Sequence[float]) -> float:
    """Return the world-Z component of the grasp frame's +X approach axis."""

    if len(gripper_pose) != 6:
        raise ValueError("gripper pose must contain x, y, z, rx, ry, rz")
    _rx, ry, rz = (float(value) for value in gripper_pose[3:])
    return -math.sin(ry)


def virtual_tip_clearance_m(
    gripper_pose: Sequence[float],
    *,
    table_z_m: float = TABLE_PLANE_Z_M,
    virtual_length_m: float = VIRTUAL_COLLISION_LENGTH_M,
    tcp_length_m: float = KINEMATIC_TCP_LENGTH_M,
) -> float:
    """Return conservative vertical clearance of the virtual distal tip."""

    if virtual_length_m < tcp_length_m:
        raise ValueError("virtual collision length cannot be shorter than the TCP")
    tip_extension_m = virtual_length_m - tcp_length_m
    tip_z_m = float(gripper_pose[2]) + tip_extension_m * approach_world_z(gripper_pose)
    return tip_z_m - float(table_z_m)


def candidate_advances_m(requested_depth_m: float) -> tuple[float, ...]:
    """Return deepest-first 5 mm insertion trials, including a seating bias."""

    requested = max(0.0, float(requested_depth_m))
    candidates = [requested + FINAL_INSERTION_BIAS_M]
    step_count = int(math.ceil(requested / FINAL_INSERTION_BIAS_M))
    candidates.extend(
        max(0.0, requested - index * FINAL_INSERTION_BIAS_M)
        for index in range(step_count + 1)
    )
    candidates.append(0.0)
    unique: list[float] = []
    for value in candidates:
        if not any(abs(value - existing) < 1e-9 for existing in unique):
            unique.append(value)
    return tuple(unique)


def meets_table_clearance(
    gripper_pose: Sequence[float],
    *,
    minimum_clearance_m: float = MINIMUM_TABLE_CLEARANCE_M,
) -> bool:
    return virtual_tip_clearance_m(gripper_pose) >= float(minimum_clearance_m)
