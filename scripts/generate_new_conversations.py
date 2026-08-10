"""
Generate 30-turn conversations for a range of prompt indices, for the main
chatbot condition only (conversations/conv_XXX.csv).

Reuses scripts/gen_convo.py's simulate_one() verbatim (same system-prompt
construction, same per-turn noise injection, same next-question generation)
so new conversations are directly comparable to the existing 500. Runs
conversations concurrently (one thread per in-flight conversation) since each
conversation's 30 turns are inherently sequential but independent across
conversations.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # autoeval_chatbot/
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

load_dotenv(BASE_DIR.parent / ".env")
load_dotenv()

if not os.getenv("PERSONAL_OPENAI_KEY"):
    raise ValueError("PERSONAL_OPENAI_KEY not found in environment/.env")

PROMPTS_DIR = BASE_DIR / "prompts"
CONV_DIR = BASE_DIR / "conversations"

TURN_LIMIT = 30

import gen_convo  # noqa: E402  (scripts/gen_convo.py)
from openai import OpenAI  # noqa: E402

_client = OpenAI(api_key=os.getenv("PERSONAL_OPENAI_KEY"))


def is_complete(path: Path) -> bool:
    """A conversation file counts as done only if it has a full 30 data rows
    (guards against files left behind by a killed/interrupted run)."""
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        return len(rows) - 1 >= TURN_LIMIT  # minus header
    except Exception:
        return False


def run_baseline(prompt_path: Path) -> Optional[Path]:
    rng = random.Random()
    style_profile = gen_convo.sample_style_profile(rng)
    return gen_convo.simulate_one(prompt_path, _client, style_profile)


def output_path_for(prompt_path: Path) -> Path:
    conv_name = prompt_path.name.replace("prompt_", "conv_").replace(".txt", ".csv")
    return CONV_DIR / conv_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    tasks: List[Path] = []
    for idx in range(args.start, args.end + 1):
        prompt_path = PROMPTS_DIR / f"prompt_{idx:03d}.txt"
        if not prompt_path.exists():
            print(f"[SKIP] {prompt_path.name} missing")
            continue
        out_path = output_path_for(prompt_path)
        if is_complete(out_path):
            continue
        tasks.append(prompt_path)

    print(f"Queued {len(tasks)} baseline conversation jobs")

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_baseline, p): p for p in tasks}
        for future in as_completed(futures):
            prompt_path = futures[future]
            try:
                result = future.result()
                completed += 1
                print(f"[{completed}/{len(tasks)}] {prompt_path.stem} -> {result}")
            except Exception as exc:
                failed += 1
                print(f"[FAILED] {prompt_path.stem}: {exc}")

    print(f"Done. completed={completed} failed={failed} total={len(tasks)}")


if __name__ == "__main__":
    main()
