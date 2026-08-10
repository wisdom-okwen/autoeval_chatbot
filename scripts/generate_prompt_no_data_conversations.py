"""
Generate conversations for the prompt_no_data ablation variant, for the 219
NEW prompt profiles only (prompt_500..prompt_718). The 131 pre-existing
English conversations for this variant were copied forward from
conversations/old/baselines/prompt_no_data/ as-is (see conversation history);
this script only needs to cover the new profiles.

Design: prompt_no_data = the REAL ShesPrEPared system prompt (shesprepared/
gpt.py's get_gpt_response), with all curated/data injections removed, but
every instruction/rule/formatting requirement left completely intact.

Implementation: rather than hand-copying gpt.py's large system-prompt string
(which drifted out of date in the old prompt_no_data/bot.py), this
monkeypatches gpt.py's four data-holding globals (decision_aid_content,
curated_reference_material, mental_health_resources,
example_sensitive_responses) to empty strings and calls the REAL,
UNMODIFIED get_gpt_response(). This guarantees the instructional text always
matches whatever gpt.py currently has -- no manual copy to keep in sync.

Like scripts/regenerate_main_chatbot.py, gpt.py's shared history.json file
mechanism is monkeypatched to per-thread, per-conversation in-memory state,
and its OPENAI_API_KEY client (out of credits) is swapped for
PERSONAL_OPENAI_KEY.

User-turn simulation reuses scripts/baseline/shared_question_generator.py
(the same dedup-aware next-question generator the old ablation conversations
used), so the new 219 stay methodologically consistent with the 131 copied
forward.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # autoeval_chatbot/
SCRIPTS_DIR = BASE_DIR / "scripts"
SHESPREPARED_DIR = BASE_DIR.parent / "shesprepared"

for p in (str(SCRIPTS_DIR), str(SCRIPTS_DIR / "baseline"), str(SHESPREPARED_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

load_dotenv(BASE_DIR.parent / ".env")
load_dotenv()

if not os.getenv("PERSONAL_OPENAI_KEY"):
    raise ValueError("PERSONAL_OPENAI_KEY not found in environment/.env")

import gen_convo  # noqa: E402  (scripts/gen_convo.py -- noise + prompt-file parsing helpers)
import gpt as shesprepared_gpt  # noqa: E402  (shesprepared/gpt.py -- the real main chatbot)
import shared_question_generator  # noqa: E402  (scripts/baseline/shared_question_generator.py)
from openai import OpenAI  # noqa: E402

PROMPTS_DIR = BASE_DIR / "prompts"
CONV_DIR = BASE_DIR / "conversations" / "new" / "baselines" / "prompt_no_data"
CONV_DIR.mkdir(parents=True, exist_ok=True)
TURN_LIMIT = 30

# ---------------------------------------------------------------------------
# Ablate DATA: blank out gpt.py's curated-content globals. Everything else in
# get_gpt_response's system prompt (instructions/rules/formatting) is untouched.
# ---------------------------------------------------------------------------
shesprepared_gpt.decision_aid_content = ""
shesprepared_gpt.curated_reference_material = ""
shesprepared_gpt.mental_health_resources = ""
shesprepared_gpt.example_sensitive_responses = ""

# shesprepared/gpt.py's own OPENAI_API_KEY account is out of credits; point
# its client at PERSONAL_OPENAI_KEY for this generation run instead.
shesprepared_gpt.client = OpenAI(api_key=os.getenv("PERSONAL_OPENAI_KEY"))

# ---------------------------------------------------------------------------
# Thread-local history instead of gpt.py's shared history.json file, so
# concurrently-simulated conversations never see each other's history.
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


def call_bot(user_input: str, max_retries: int = 5) -> str:
    reply = ""
    for attempt in range(max_retries):
        reply = shesprepared_gpt.get_gpt_response(user_input)
        if not reply.startswith("GPT experienced an internal error"):
            return reply
        hist = getattr(_thread_local, "history", [])
        if hist and hist[-1].get("bot") == reply:
            hist.pop()
        time.sleep((2 ** attempt) + random.random())
    return reply


# ---------------------------------------------------------------------------
# manifest for safe resumability
# ---------------------------------------------------------------------------
MANIFEST_DIR = Path(
    os.environ.get(
        "REGEN_MANIFEST_DIR",
        "/tmp/claude-365345/-playpen-ssd-wokwen/456462f0-4b43-46e5-a59b-ee329fa35f6b/scratchpad/prompt_no_data_manifest",
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
_STARTER_PATTERN = re.compile(r"You may start by asking:\s*\n?\"(.+?)\"", re.MULTILINE)
_FOLLOWUP_PATTERN = re.compile(r"A possible follow-up you might ask later:\n\"(.+?)\"", re.MULTILINE)


def _load_starter_and_guidance(prompt_path: Path):
    text = prompt_path.read_text(encoding="utf-8")
    starter_match = _STARTER_PATTERN.search(text)
    if not starter_match:
        raise ValueError(f"Starter question not found in {prompt_path.name}")
    starter = starter_match.group(1).strip()
    followup_match = _FOLLOWUP_PATTERN.search(text)
    follow = (
        followup_match.group(1).strip()
        if followup_match
        else "ask something more specific about access, cost, side effects, adherence, or stigma."
    )
    return starter, follow


def _load_persona_and_language(prompt_path: Path):
    text = prompt_path.read_text(encoding="utf-8")
    persona_lines = []
    for line in text.splitlines():
        if line.strip().startswith("You may start by asking:"):
            break
        persona_lines.append(line.strip())
    persona = " ".join(persona_lines[-40:])
    json_path = prompt_path.with_suffix(".json")
    language = "English"
    if json_path.exists():
        import json

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            language = data.get("profile", {}).get("language", language)
        except json.JSONDecodeError:
            pass
    return persona, language


def _generate_next_question(previous_question, bot_reply, starter, follow_hint, persona, asked_history) -> str:
    history_list = list(asked_history)
    history_set = {q.lower() for q in history_list}
    prompt = (
        "Generate the NEXT user question only (no prefix) continuing a 30-turn conversation about HIV prevention/PrEP. "
        "User never thanks explicitly, keeps direct, informal, authentic. Avoid repeating earlier phrasing. "
        f"Persona context: {persona}\n"
        f"Previous user question: {previous_question}\n"
        f"Assistant reply: {bot_reply}\n"
        f"Starter question was: {starter}\n"
        f"Follow-up hint: {follow_hint}\n"
        "Questions already asked so far (do not repeat or restate these ideas):\n"
        + "\n".join(f"- {q}" for q in history_list)
        + "\n"
        "Constraints: under 25 words, no greetings, no closing, no lists, one sentence. Output only the question."
    )
    candidate = ""
    for _ in range(5):
        candidate = shared_question_generator.generate_next_question(prompt, tuple(history_list)).strip()
        candidate = re.sub(r"\s+", " ", candidate)
        lower = candidate.lower()
        if lower not in history_set and candidate:
            return candidate
        prompt += "\nPlease provide a different question that has not been asked."
    return candidate + " ?"


def simulate_one(prompt_path: Path) -> Path:
    _thread_local.history = []  # reset -- ThreadPoolExecutor reuses worker threads

    starter, follow_hint = _load_starter_and_guidance(prompt_path)
    persona, language = _load_persona_and_language(prompt_path)

    conversation_file = CONV_DIR / prompt_path.name.replace("prompt_", "conv_").replace(".txt", ".csv")

    asked_history: List[str] = []  # strictly-earlier-turn questions only
    current_question = starter
    with conversation_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["turn", "user", "bot", "language", "has_error"])
        for turn in range(1, TURN_LIMIT + 1):
            bot_reply = call_bot(current_question)
            writer.writerow([turn, current_question, bot_reply, language, 0])
            asked_history.append(current_question)
            if turn == TURN_LIMIT:
                break
            next_question = _generate_next_question(
                current_question, bot_reply, starter, follow_hint, persona, asked_history
            )
            current_question = next_question

    return conversation_file


def run_one(idx: int) -> Optional[Path]:
    prompt_path = PROMPTS_DIR / f"prompt_{idx:03d}.txt"
    if not prompt_path.exists():
        print(f"[SKIP] prompt_{idx:03d}.txt missing")
        return None
    result = simulate_one(prompt_path)
    mark_done(idx)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    tasks = [idx for idx in range(args.start, args.end + 1) if not is_done(idx)]
    print(f"Queued {len(tasks)} prompt_no_data conversations")

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
