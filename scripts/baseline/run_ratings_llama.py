#!/usr/bin/env python3
"""Run LLaMA rating pipelines for baseline conversation outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from _ratings_common import BASELINES, infer_range, run_module_main, temporary_attrs


def run_llama_ratings(convo_dir: Path, llama_root: Path, start_idx: int, end_idx: int) -> None:
    """Run LLaMA-based rating scripts against a conversation directory."""
    from scripts.ratings.llama import overall_api as llama_overall
    from scripts.ratings.llama import persona_api as llama_persona
    from scripts.ratings.llama import perturn_api as llama_perturn

    project_root = convo_dir.parent

    jobs = [
        (llama_overall, llama_root / "overall", ["--start", str(start_idx), "--end", str(end_idx)]),
        (llama_persona, llama_root / "persona", ["--start", str(start_idx), "--end", str(end_idx)]),
        (llama_perturn, llama_root / "per_turn", ["--start", str(start_idx), "--end", str(end_idx)]),
    ]

    for module, output_dir, cli_args in jobs:
        output_dir.mkdir(parents=True, exist_ok=True)
        attr_changes = [
            (module, "CONVERSATIONS_DIR", convo_dir),
            (module, "PROJECT_ROOT", project_root),
            (module, "RATINGS_DIR", output_dir),
        ]
        try:
            with temporary_attrs(attr_changes):
                run_module_main(module, cli_args)
        except SystemExit as exc:  # Allow module to exit with non-zero status
            if exc.code not in (0, None):
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLaMA ratings for baseline conversation sets")
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
        llama_root = variant_root / "ratings" / "llama"

        print(f"\n=== {variant} :: conversations {start_idx:03d}-{end_idx:03d} ===")
        llama_root.mkdir(parents=True, exist_ok=True)

        print("-> Running LLaMA ratings")
        run_llama_ratings(convo_dir, llama_root, start_idx, end_idx)

    print("\nLLaMA ratings pipeline completed.")


if __name__ == "__main__":
    main()
