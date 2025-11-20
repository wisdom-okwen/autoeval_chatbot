"""Utility to generate follow-up user questions for baseline runs."""
from __future__ import annotations

import os
import random
import re
from typing import Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_API_KEY = (
    os.getenv("PERSONAL_OPENAI_KEY")
)
if not _API_KEY:
    raise ValueError("Missing OPENAI_API_KEY or PERSONAL_OPENAI_KEY in environment.")

_client = OpenAI(api_key=_API_KEY)

_SYSTEM_PROMPT = (
    "You craft the next user question in an ongoing PrEP counseling chat."
    " Stay informal yet respectful. Do NOT repeat earlier questions even if wording changes."
    " Avoid greetings or thanks. Keep it under 25 words, one sentence, no lists."
)

_NOISY_PROPORTION = 0.4
_RNG = random.Random()

_QWERTY_NEIGHBORS = {
    "a": "sqwz",
    "b": "vghn",
    "c": "xdfv",
    "d": "ersfcx",
    "e": "wsdr",
    "f": "drtgcv",
    "g": "ftyhbv",
    "h": "gyujnb",
    "i": "ujko",
    "j": "huikmn",
    "k": "jiolm,",
    "l": "kop;.",
    "m": "njk,",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol;[",
    "q": "wsa",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tugh",
    "z": "asx",
}

_LEET_MAP = {"o": "0", "i": "1", "e": "3", "a": "4", "s": "5", "t": "7"}
_FILLERS = ["uh", "um", "like", "you know", "idk", "i mean"]


def _qwerty_typo(ch: str) -> str:
    if ch.lower() in _QWERTY_NEIGHBORS and _RNG.random() < 0.6:
        return _RNG.choice(_QWERTY_NEIGHBORS[ch.lower()])
    return ch


def _apply_leet(word: str, probability: float = 0.2) -> str:
    out = []
    for ch in word:
        if ch.lower() in _LEET_MAP and _RNG.random() < probability:
            out.append(_LEET_MAP[ch.lower()])
        else:
            out.append(ch)
    return "".join(out)


def _corrupt_word(
    word: str,
    typo_rate: float,
    swap_rate: float,
    drop_rate: float,
    dup_rate: float,
    leet_rate: float,
) -> str:
    if not word:
        return word
    w = word
    if len(w) > 3 and _RNG.random() < drop_rate:
        i = _RNG.randrange(len(w))
        w = w[:i] + w[i + 1 :]
    if len(w) > 3 and _RNG.random() < swap_rate:
        i = _RNG.randrange(len(w) - 1)
        w = w[:i] + w[i + 1] + w[i] + w[i + 2 :]
    if _RNG.random() < typo_rate:
        i = _RNG.randrange(len(w))
        w = w[:i] + _qwerty_typo(w[i]) + w[i + 1 :]
    if len(w) > 1 and _RNG.random() < dup_rate:
        i = _RNG.randrange(len(w))
        w = w[:i] + w[i] * 2 + w[i :]
    if _RNG.random() < leet_rate:
        w = _apply_leet(w, probability=0.8)
    return w


def _sprinkle_spacing(text: str, double_space_prob: float, runon_prob: float) -> str:
    if _RNG.random() < double_space_prob:
        text = re.sub(r"\s", lambda m: "  " if _RNG.random() < 0.3 else m.group(0), text)
    if _RNG.random() < runon_prob:
        text = text.replace(",", "", 1).replace("?", "", 1)
    return text


def _add_fillers(text: str, filler_rate: float) -> str:
    parts = text.split()
    if parts and _RNG.random() < filler_rate:
        pos = _RNG.randrange(len(parts))
        parts.insert(pos, _RNG.choice(_FILLERS))
    return " ".join(parts)


def _sample_style_profile() -> Tuple[float, ...]:
    return (
        _RNG.uniform(0.05, 0.12),  # typo_rate
        _RNG.uniform(0.02, 0.06),  # swap_rate
        _RNG.uniform(0.02, 0.05),  # drop_rate
        _RNG.uniform(0.01, 0.03),  # dup_rate
        _RNG.uniform(0.05, 0.12),  # leet_rate
        _RNG.uniform(0.15, 0.35),  # double_space_prob
        _RNG.uniform(0.1, 0.25),   # runon_prob
        _RNG.uniform(0.2, 0.45),   # filler_rate
    )


def _apply_noise(question: str) -> str:
    typo_rate, swap_rate, drop_rate, dup_rate, leet_rate, double_space_prob, runon_prob, filler_rate = (
        _sample_style_profile()
    )
    body = question.rstrip().rstrip("?")
    words = [
        _corrupt_word(word, typo_rate, swap_rate, drop_rate, dup_rate, leet_rate)
        for word in body.split()
    ]
    noisy = " ".join(words)
    noisy = noisy.lower() if _RNG.random() < 0.25 else noisy
    noisy = _sprinkle_spacing(noisy, double_space_prob, runon_prob)
    noisy = _add_fillers(noisy, filler_rate)
    noisy = noisy.strip()
    if not noisy:
        return question
    if not noisy.endswith("?"):
        noisy = noisy.rstrip(".!") + "?"
    return noisy


def generate_next_question(prompt: str, history: Tuple[str, ...]) -> str:
    """Return the next user utterance, considering prior questions."""
    history_text = "\n".join(f"- {q}" for q in history)
    completion = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Previously asked questions (do not repeat or paraphrase):\n{history_text}\n\n"
                    f"Instruction:\n{prompt}"
                ),
            },
        ],
        temperature=0.8,
        max_tokens=64,
    )
    question = completion.choices[0].message.content.strip()
    question = re.sub(r"\s+", " ", question)
    if question and not question.endswith("?"):
        question = question.rstrip(".!") + "?"
    if question and _RNG.random() < _NOISY_PROPORTION:
        # Mimic imperfect typing on a subset of the questions.
        question = _apply_noise(question)
    return question
