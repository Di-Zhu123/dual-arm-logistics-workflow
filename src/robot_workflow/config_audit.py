"""Read-only checks for whether a camera rig is safe for 3D fusion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CameraRigAudit:
    device_ids: tuple[int, ...]
    missing_fields: tuple[str, ...]

    @property
    def fusion_ready(self) -> bool:
        return not self.missing_fields


def audit_robotdata_config(config: Mapping[str, Any]) -> CameraRigAudit:
    missing: list[str] = []
    ids: list[int] = []
    for arm in ("left_arm_config", "right_arm_config"):
        section = config.get(arm, {})
        if "device_id" not in section:
            missing.append(f"{arm}.device_id")
        else:
            ids.append(int(section["device_id"]))
        if "cam_extrinsic" not in section:
            missing.append(f"{arm}.cam_extrinsic")
    head = config.get("head_camera_config", {})
    if "device_id" not in head:
        missing.append("head_camera_config.device_id")
    else:
        ids.append(int(head["device_id"]))
    if "world_from_camera" not in head:
        missing.append("head_camera_config.world_from_camera")
    if len(ids) != len(set(ids)):
        missing.append("camera device_ids must be unique")
    return CameraRigAudit(tuple(ids), tuple(missing))


def audit_robotdata_config_file(path: str | Path) -> CameraRigAudit:
    return audit_robotdata_config(json.loads(Path(path).read_text("utf-8")))
