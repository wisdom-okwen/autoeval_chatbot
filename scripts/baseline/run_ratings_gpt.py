#!/usr/bin/env python3
"""Run GPT rating pipelines for baseline conversation outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from _ratings_common import BASELINES, PROJECT_ROOT, infer_range, run_module_main, temporary_attrs


def run_gpt_ratings(convo_dir: Path, gpt_root: Path, start_idx: int, end_idx: int) -> None:
    """Run GPT-based rating scripts against a conversation directory."""
    gpt_dir = PROJECT_ROOT / "scripts" / "ratings" / "gpt"
    if str(gpt_dir) not in sys.path:
        sys.path.insert(0, str(gpt_dir))

    from scripts.ratings.gpt import utils as gpt_utils
    from scripts.ratings.gpt import overall as gpt_overall
    from scripts.ratings.gpt import persona as gpt_persona
    from scripts.ratings.gpt import perturn as gpt_perturn

    utils_modules: Iterable[object] = {gpt_utils}
    utils_alias = sys.modules.get("utils")
    if utils_alias is not None:
        utils_modules = {gpt_utils, utils_alias}

    gpt_root.mkdir(parents=True, exist_ok=True)

    attr_changes = []
    for module in utils_modules:
        attr_changes.extend([
            (module, "CONVERSATIONS_DIR", convo_dir),
            (module, "RATINGS_OUTPUT_DIR", gpt_root),
        ])

    attr_changes.extend([
        (gpt_overall, "RATINGS_OUTPUT_DIR", gpt_root),
        (gpt_persona, "RATINGS_OUTPUT_DIR", gpt_root),
        (gpt_perturn, "RATINGS_OUTPUT_DIR", gpt_root),
    ])

    with temporary_attrs(attr_changes):
        run_module_main(gpt_overall, ["--start", str(start_idx), "--end", str(end_idx)])
        run_module_main(gpt_persona, ["--start", str(start_idx), "--end", str(end_idx)])
        run_module_main(gpt_perturn, ["--start", str(start_idx), "--end", str(end_idx)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT ratings for baseline conversation sets")
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(BASELINES.keys()),
        help="Baseline variant(s) to rate (default: all)",
    )
    parser.add_argument("--start", type=int, help="Override start conversation index")
    parser.add_argument("--end", type=int, help="Override end conversation index")
    args = parser.parse_args()

    variants = args.variant or list(BASELINES.keys())

    for variant in variants:
        config = BASELINES[variant]
        convo_dir: Path = config["convo_dir"]
        if not convo_dir.exists():
            print(f"[WARN] Conversation directory missing for {variant}: {convo_dir}")
            continue

        try:
            auto_start, auto_end = infer_range(convo_dir)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue

        start_idx = args.start if args.start is not None else auto_start
        end_idx = args.end if args.end is not None else auto_end

        variant_root = convo_dir.parent
        gpt_root = variant_root / "ratings" / "gpt"

        print(f"\n=== {variant} :: conversations {start_idx:03d}-{end_idx:03d} ===")
        gpt_root.mkdir(parents=True, exist_ok=True)

        print("-> Running GPT ratings")
        run_gpt_ratings(convo_dir, gpt_root, start_idx, end_idx)

    print("\nGPT ratings pipeline completed.")


if __name__ == "__main__":
    main()
