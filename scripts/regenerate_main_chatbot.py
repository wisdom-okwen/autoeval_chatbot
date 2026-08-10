"""
Regenerate ALL main-chatbot (system 1) conversations
(conversations/conv_000..conv_718.csv) using the REAL ShesPrEPared bot
(shesprepared/gpt.py's get_gpt_response), instead of the generic placeholder
prompt scripts/gen_convo.py used before.

NOTE ON TERMINOLOGY: this is the MAIN CHATBOT / system 1 (the one real users
talk to). It is NOT one of "the baselines" -- those are the two ablation
variants (prompt_no_data, data_no_prompt) that live under
conversations/baselines/ and are handled by separate scripts.

Bot logic (system prompt, curated data, formatting rules) is reused
UNCHANGED by importing shesprepared/gpt.py directly and calling its
get_gpt_response(). That module normally persists conversation history to a
single shared history.json file, which is wrong for simulating hundreds of
independent conversations concurrently -- so its load_history/save_history
are monkeypatched here to use per-thread, per-conversation in-memory state
instead. Everything else (system prompt text, model, temperature, max_tokens)
is exactly what production uses.

User-turn simulation (next-question generation, per-turn typo/noise
injection) reuses scripts/gen_convo.py's existing helpers, unchanged, so the
"user" side of these conversations stays methodologically consistent with
the rest of the dataset -- only the bot's response logic is now the real
system.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # autoeval_chatbot/
SCRIPTS_DIR = BASE_DIR / "scripts"
SHESPREPARED_DIR = BASE_DIR.parent / "shesprepared"

for p in (str(SCRIPTS_DIR), str(SHESPREPARED_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

load_dotenv(BASE_DIR.parent / ".env")
load_dotenv()

if not os.getenv("PERSONAL_OPENAI_KEY"):
    raise ValueError("PERSONAL_OPENAI_KEY not found in environment/.env (needed for user-turn simulation and, "
                     "since shesprepared's own OPENAI_API_KEY account is out of credits, for the main chatbot too)")

import gen_convo  # noqa: E402  (scripts/gen_convo.py -- noise + next-question helpers)
import gpt as shesprepared_gpt  # noqa: E402  (shesprepared/gpt.py -- the real main chatbot)
from openai import OpenAI  # noqa: E402

PROMPTS_DIR = BASE_DIR / "prompts"
CONV_DIR = BASE_DIR / "conversations" / "new"
CONV_DIR.mkdir(parents=True, exist_ok=True)
TURN_LIMIT = 30

_userturn_client = OpenAI(api_key=os.getenv("PERSONAL_OPENAI_KEY"))

# ---------------------------------------------------------------------------
# Make shesprepared_gpt's history thread-local instead of shared-file-backed,
# so concurrently-simulated conversations never see each other's history.
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def _local_load_history():
    return list(getattr(_thread_local, "history", []))


def _local_save_history(user, bot):
    hist = getattr(_thread_local, "history", [])
    hist.append({"user": user, "bot": bot})
    _thread_local.history = hist


shesprepared_gpt.load_history = _local_load_history
shesprepared_gpt.save_history = _local_save_history

# shesprepared/gpt.py's own OPENAI_API_KEY account is out of credits; point
# its client at PERSONAL_OPENAI_KEY for this generation run instead (no
# changes to the actual production shesprepared app files).
shesprepared_gpt.client = OpenAI(api_key=os.getenv("PERSONAL_OPENAI_KEY"))


def call_main_chatbot(user_input: str, max_retries: int = 5) -> str:
    """Call the real, unmodified get_gpt_response, retrying past the
    internal-error string it returns on API failures (it swallows exceptions
    instead of raising)."""
    reply = ""
    for attempt in range(max_retries):
        reply = shesprepared_gpt.get_gpt_response(user_input)
        if not reply.startswith("GPT experienced an internal error"):
            return reply
        # strip the bogus entry save_history() just appended before retrying
        hist = getattr(_thread_local, "history", [])
        if hist and hist[-1].get("bot") == reply:
            hist.pop()
        time.sleep((2 ** attempt) + random.random())
    return reply


# ---------------------------------------------------------------------------
# manifest for safe resumability across (possibly interrupted) runs of THIS
# regeneration job specifically -- a completed file from the OLD generic
# prompt still has 30 rows, so row-count alone can't tell us it needs redoing.
# ---------------------------------------------------------------------------
MANIFEST_DIR = Path(
    os.environ.get(
        "REGEN_MANIFEST_DIR",
        "/tmp/claude-365345/-playpen-ssd-wokwen/456462f0-4b43-46e5-a59b-ee329fa35f6b/scratchpad/main_chatbot_regen_manifest",
    )
)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def is_done(idx: int) -> bool:
    return (MANIFEST_DIR / f"{idx:03d}.done").exists()


def mark_done(idx: int) -> None:
    (MANIFEST_DIR / f"{idx:03d}.done").touch()


# ---------------------------------------------------------------------------
# conversation simulation
# ---------------------------------------------------------------------------
def simulate_one(prompt_path: Path) -> Path:
    _thread_local.history = []  # reset -- ThreadPoolExecutor reuses worker threads

    text = prompt_path.read_text(encoding="utf-8")
    profile, context = gen_convo.extract_profile_and_context(text)
    first_q = gen_convo.extract_first_question(text)
    if not first_q:
        raise ValueError(f"No start question found in {prompt_path.name}")

    conv_path = CONV_DIR / (prompt_path.stem.replace("prompt", "conv") + ".csv")
    rng = random.Random()
    style_profile = gen_convo.sample_style_profile(rng)
    persona_language = profile.get("language") or profile.get("Language") or "English"
    noisy_turns = set(random.sample(range(1, TURN_LIMIT + 1), k=int(round(TURN_LIMIT * gen_convo.NOISY_PROPORTION))))

    with conv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["turn", "user", "bot", "language", "has_error"])

        clean_user = first_q
        last_pairs = []
        for t in range(1, TURN_LIMIT + 1):
            has_error = t in noisy_turns
            noisy_user = gen_convo.apply_style_profile(clean_user, style_profile) if has_error else clean_user

            bot_reply = call_main_chatbot(noisy_user)
            w.writerow([t, noisy_user, bot_reply, persona_language, 1 if has_error else 0])

            last_pairs.append((clean_user, bot_reply))
            last_pairs = last_pairs[-3:]
            if t == TURN_LIMIT:
                break
            clean_user = gen_convo.next_user_question_clean(
                _userturn_client, last_pairs[-1][0], last_pairs[-1][1], profile, context
            )

    return conv_path


def run_one(idx: int) -> Optional[Path]:
    prompt_path = PROMPTS_DIR / f"prompt_{idx:03d}.txt"
    if not prompt_path.exists():
        print(f"[SKIP] prompt_{idx:03d}.txt missing")
        return None
    result = simulate_one(prompt_path)
    mark_done(idx)
    return result


def is_english(idx: int) -> bool:
    json_path = PROMPTS_DIR / f"prompt_{idx:03d}.json"
    if not json_path.exists():
        return False
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("profile", {}).get("language") == "English"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Process every prompt in range, not just English ones (default: English-only, to limit API cost).",
    )
    args = parser.parse_args()

    in_range = range(args.start, args.end + 1)
    if args.all_languages:
        candidates = list(in_range)
    else:
        candidates = [idx for idx in in_range if is_english(idx)]
        skipped_non_english = (args.end - args.start + 1) - len(candidates)
        print(f"English-only mode: {len(candidates)} English prompts in range (skipping {skipped_non_english} non-English)")

    tasks = [idx for idx in candidates if not is_done(idx)]
    print(f"Queued {len(tasks)} main-chatbot conversations for regeneration with the real ShesPrEPared bot")

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, idx): idx for idx in tasks}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                completed += 1
                print(f"[{completed}/{len(tasks)}] conv_{idx:03d} -> {result}")
            except Exception as exc:
                failed += 1
                print(f"[FAILED] conv_{idx:03d}: {exc}")

    print(f"Done. completed={completed} failed={failed} total={len(tasks)}")


if __name__ == "__main__":
    main()
