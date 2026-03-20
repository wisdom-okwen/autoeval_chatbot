#!/usr/bin/env python3
"""Shared helpers for baseline rating runners."""
from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Sequence, Tuple

BASELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASELINES = {
    "data_no_prompt": {
        "convo_dir": BASELINE_ROOT / "data_no_prompt" / "convo",
    },
    "prompt_no_data": {
        "convo_dir": BASELINE_ROOT / "prompt_no_data" / "convo",
    },
}

CONV_FILENAME_RE = re.compile(r"conv_(\d+)\.csv$")

Change = Tuple[object, str, object]


@contextmanager
def temporary_attrs(changes: Sequence[Change]):
    """Temporarily set attributes on modules or objects."""
    originals: List[Change] = []
    try:
        for target, attr, new_value in changes:
            originals.append((target, attr, getattr(target, attr)))
            setattr(target, attr, new_value)
        yield
    finally:
        for target, attr, old_value in reversed(originals):
            setattr(target, attr, old_value)


def infer_range(convo_dir: Path) -> tuple[int, int]:
    """Infer min/max conversation indices present in a directory."""
    indices: List[int] = []
    for path in convo_dir.glob("conv_*.csv"):
        match = CONV_FILENAME_RE.search(path.name)
        if match:
            indices.append(int(match.group(1)))
    if not indices:
        raise ValueError(f"No conversation CSV files found in {convo_dir}")
    return min(indices), max(indices)


def run_module_main(module, args: Sequence[str]) -> None:
    """Execute a rating module's main() with injected CLI args."""
    argv_backup = sys.argv
    sys.argv = [module.__file__ or module.__name__] + list(args)
    try:
        module.main()
    finally:
        sys.argv = argv_backup
