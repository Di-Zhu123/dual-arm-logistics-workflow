#!/usr/bin/env python3
"""Fast whole-object workflow for placing an orange medicine box in the green box."""

from __future__ import annotations

import sys

from run_pistol_grasp_workflow import main


if __name__ == "__main__":
    sys.argv.extend(
        [
            "--target-text",
            "orange medicine box",
            "--whole-object-grasp",
            "--fast-magazine",
            "--relaxed-grasp-quality",
            "--grasp-planning-attempts",
            "1",
        ]
    )
    raise SystemExit(main())
