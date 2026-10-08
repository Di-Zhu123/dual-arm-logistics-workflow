#!/usr/bin/env python3
"""Dedicated entry point for grasping a magazine and placing it in the green box."""

from __future__ import annotations

import sys

from run_pistol_grasp_workflow import main


if __name__ == "__main__":
    sys.argv.extend(
        [
            "--target-text",
            "rectangular black object",
            "--whole-object-grasp",
            "--fast-magazine",
            "--relaxed-grasp-quality",
            "--grasp-planning-attempts",
            "1",
        ]
    )
    raise SystemExit(main())
