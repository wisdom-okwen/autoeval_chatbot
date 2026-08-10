"""
Generate conversations for the data_no_prompt ablation variant, for the 219
NEW prompt profiles only (prompt_500..prompt_718). The 131 pre-existing
English conversations for this variant were copied forward from
conversations/old/baselines/data_no_prompt/ as-is (see conversation history);
this script only needs to cover the new profiles.

Design: data_no_prompt = ALL of the real ShesPrEPared curated reference data
(shesprepared/gpt.py's decision_aid_content, curated_reference_material,
mental_health_resources, example_sensitive_responses -- imported directly
from gpt.py so it is always byte-identical/up-to-date with production, unlike
the old data_no_prompt/bot.py which had hand-copied a stale subset), wrapped
in a genuinely MINIMAL system prompt (none of gpt.py's elaborate
instructions/formatting rules).

Unlike prompt_no_data (which reuses gpt.py's get_gpt_response unmodified),
this variant needs its own response function since the wrapper text itself
must change -- so history is passed explicitly as an argument rather than
via gpt.py's load_history()/save_history() monkeypatching.

User-turn simulation reuses scripts/baseline/shared_question_generator.py
(the same dedup-aware next-question generator the old ablation conversations
used), so the new 219 stay methodologically consistent with the 131 copied
forward.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

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

import gen_convo  # noqa: E402  (scripts/gen_convo.py -- noise + prompt-file parsing helpers, unused here but kept for parity)
import gpt as shesprepared_gpt  # noqa: E402  (shesprepared/gpt.py -- imported ONLY for its real, current data globals)
import shared_question_generator  # noqa: E402  (scripts/baseline/shared_question_generator.py)
from openai import OpenAI  # noqa: E402

PROMPTS_DIR = BASE_DIR / "prompts"
CONV_DIR = BASE_DIR / "conversations" / "new" / "baselines" / "data_no_prompt"
CONV_DIR.mkdir(parents=True, exist_ok=True)
TURN_LIMIT = 30
HISTORY_LENGTH = shesprepared_gpt.HISTORY_LENGTH  # 4, matches production

_client = OpenAI(api_key=os.getenv("PERSONAL_OPENAI_KEY"))

_MINIMAL_SYSTEM_PREAMBLE = (
    "You are ShesPrEPared, an assistant that answers questions about HIV prevention and PrEP "
    "using the reference material below.\n\n"
)


def _build_system_prompt(history_pairs: List[Dict[str, str]]) -> str:
    formatted_history = "\n".join(
        f"User: {h['user']}\nPrEPBot: {h['bot']}" for h in history_pairs
    ) if history_pairs else ""
    return (
        _MINIMAL_SYSTEM_PREAMBLE
        + f"{shesprepared_gpt.decision_aid_content}\n\n"
        + f"{shesprepared_gpt.curated_reference_material}\n\n"
        + f"{shesprepared_gpt.mental_health_resources}\n\n"
        + f"{shesprepared_gpt.example_sensitive_responses}\n\n"
        + f"Consider the following conversation history as additional context: {formatted_history}.\n\n"
        + "Answer the user's question below."
    )


def call_bot(user_input: str, history_pairs: List[Dict[str, str]], max_retries: int = 5) -> str:
    system_prompt = _build_system_prompt(history_pairs)
    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=350,
                temperature=0.75,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            time.sleep((2 ** attempt) + random.random())
    return "[bot error: exhausted retries]"


# ---------------------------------------------------------------------------
# manifest for safe resumability
# ---------------------------------------------------------------------------
MANIFEST_DIR = Path(
    os.environ.get(
        "REGEN_MANIFEST_DIR",
        "/tmp/claude-365345/-playpen-ssd-wokwen/456462f0-4b43-46e5-a59b-ee329fa35f6b/scratchpad/data_no_prompt_manifest",
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
    starter, follow_hint = _load_starter_and_guidance(prompt_path)
    persona, language = _load_persona_and_language(prompt_path)

    conversation_file = CONV_DIR / prompt_path.name.replace("prompt_", "conv_").replace(".txt", ".csv")

    asked_history: List[str] = []
    history_pairs: List[Dict[str, str]] = []  # last HISTORY_LENGTH {user, bot} pairs, like production
    current_question = starter
    with conversation_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["turn", "user", "bot", "language", "has_error"])
        for turn in range(1, TURN_LIMIT + 1):
            bot_reply = call_bot(current_question, history_pairs[-HISTORY_LENGTH:])
            writer.writerow([turn, current_question, bot_reply, language, 0])

            asked_history.append(current_question)
            history_pairs.append({"user": current_question, "bot": bot_reply})

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
    print(f"Queued {len(tasks)} data_no_prompt conversations")

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
