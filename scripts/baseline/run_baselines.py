"""Generate baseline conversations for both configurations using shared prompts."""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
PROMPTS_DIR = BASE_DIR / "prompts"

BASELINES: Dict[str, Dict[str, Path]] = {
    "data_no_prompt": {
        "module": Path("baseline.data_no_prompt.bot"),
        "convo_dir": BASE_DIR / "baseline" / "data_no_prompt" / "convo",
    },
    "prompt_no_data": {
        "module": Path("baseline.prompt_no_data.bot"),
        "convo_dir": BASE_DIR / "baseline" / "prompt_no_data" / "convo",
    },
}

TURN_LIMIT = 30

_STARTER_PATTERN = re.compile(r"You may start by asking:\s*\n?\"(.+?)\"", re.MULTILINE)
_FOLLOWUP_PATTERN = re.compile(r"A possible follow-up you might ask later:\n\"(.+?)\"", re.MULTILINE)


def _list_prompt_files() -> Iterable[Path]:
    for path in sorted(PROMPTS_DIR.glob("prompt_*.txt")):
        if path.is_file():
            yield path


def _load_starter_and_guidance(prompt_path: Path) -> Tuple[str, str]:
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


def _load_persona_and_language(prompt_path: Path) -> Tuple[str, str]:
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


def _generate_next_question(
    previous_question: str,
    bot_reply: str,
    starter: str,
    follow_hint: str,
    persona: str,
    asked_history: Iterable[str],
    generator: "QuestionGenerator",
) -> str:
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
    for attempt in range(5):
        candidate = generator(prompt, tuple(history_list)).strip()
        candidate = re.sub(r"\s+", " ", candidate)
        lower = candidate.lower()
        if lower not in history_set and candidate:
            return candidate
        prompt += (
            "\nPlease provide a different question that has not been asked."
        )
    return candidate + " ?"


QuestionGenerator = Callable[[str, Tuple[str, ...]], str]


def _ensure_question_generator_loaded() -> QuestionGenerator:
    try:
        module = import_module("baseline.shared_question_generator")
    except ModuleNotFoundError:
        raise RuntimeError(
            "baseline.shared_question_generator module is required. Create it before running baselines."
        )
    if not hasattr(module, "generate_next_question"):
        raise RuntimeError(
            "shared_question_generator must expose generate_next_question(prompt: str, history: Tuple[str, ...]) -> str"
        )
    return getattr(module, "generate_next_question")


def simulate_for_variant(variant: str, get_response: Callable[[str], str],
                         convo_dir: Path) -> None:
    convo_dir.mkdir(parents=True, exist_ok=True)
    question_generator = _ensure_question_generator_loaded()
    for prompt_path in _list_prompt_files():
        starter, follow_hint = _load_starter_and_guidance(prompt_path)
        persona, language = _load_persona_and_language(prompt_path)
        conversation_file = convo_dir / prompt_path.name.replace("prompt_", "conv_").replace(".txt", ".csv")
        if conversation_file.exists():
            continue
        asked_set = {starter.lower()}
        asked_history = [starter]
        current_question = starter
        with conversation_file.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["turn", "user", "bot", "language", "has_error"])
            for turn in range(1, TURN_LIMIT + 1):
                history_block = "\n".join(f"- {q}" for q in asked_history)
                bot_input = (
                    f"Persona: {persona}\n"
                    f"Previously asked questions (avoid repeating these):\n{history_block}\n"
                    f"Current user question: {current_question}\n"
                    "If this question overlaps with earlier ones, gently remind the user to ask something different."
                )
                bot_reply = get_response(bot_input)
                writer.writerow([turn, current_question, bot_reply, language, 0])
                if turn == TURN_LIMIT:
                    break
                next_question = _generate_next_question(
                    current_question,
                    bot_reply,
                    starter,
                    follow_hint,
                    persona,
                    asked_history,
                    question_generator,
                )
                asked_set.add(next_question.lower())
                asked_history.append(next_question)
                current_question = next_question


def main() -> None:
    def _run_variant(name: str, config: Dict[str, Path]) -> None:
        module_path = str(config["module"])
        module = import_module(module_path)
        if not hasattr(module, "get_response"):
            raise RuntimeError(f"{module_path} must define get_response()")
        get_response = getattr(module, "get_response")
        convo_dir = config["convo_dir"]
        print(f"[{name}] generating conversations -> {convo_dir}")
        simulate_for_variant(name, get_response, convo_dir)
        print(f"[{name}] completed")

    with ThreadPoolExecutor(max_workers=len(BASELINES)) as executor:
        futures = {
            executor.submit(_run_variant, variant, config): variant
            for variant, config in BASELINES.items()
        }
        for future in as_completed(futures):
            variant = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[{variant}] failed: {exc}")
                raise


if __name__ == "__main__":
    main()
