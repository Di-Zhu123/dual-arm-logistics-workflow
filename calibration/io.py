"""Calibration dataset I/O and schema validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .board import BoardSpec
from .geometry import validate_transform


SCHEMA_VERSION = 1


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    target.write_text(encoded, "utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(root: str | Path, *, expected_mode: str | None = None) -> tuple[Path, dict[str, Any], BoardSpec]:
    directory = Path(root).resolve()
    manifest = read_json(directory / "dataset.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported calibration dataset schema")
    if expected_mode is not None and manifest.get("mode") != expected_mode:
        raise ValueError(
            f"dataset mode must be {expected_mode}, got {manifest.get('mode')}"
        )
    board = BoardSpec.from_mapping(manifest["board"])
    return directory, manifest, board


def load_sample(sample_directory: str | Path) -> dict[str, Any]:
    directory = Path(sample_directory)
    sample = read_json(directory / "sample.json")
    if sample.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported sample schema: {directory}")
    for camera_name, camera in sample.get("cameras", {}).items():
        rgb = directory / camera["rgb_file"]
        depth = directory / camera["depth_file"]
        if not rgb.is_file() or not depth.is_file():
            raise ValueError(f"{camera_name} camera files are missing in {directory}")
        if camera.get("rgb_sha256") != sha256_file(rgb):
            raise ValueError(f"{camera_name} RGB digest mismatch in {directory}")
        if camera.get("depth_sha256") != sha256_file(depth):
            raise ValueError(f"{camera_name} depth digest mismatch in {directory}")
    if "base_from_tool" in sample.get("robot", {}):
        validate_transform(sample["robot"]["base_from_tool"], name="base_from_tool")
    return sample


def sample_directories(root: Path) -> list[Path]:
    return sorted(
        path for path in (root / "samples").glob("sample_*") if path.is_dir()
    )

