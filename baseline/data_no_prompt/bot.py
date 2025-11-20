"""Baseline chatbot variant: minimal instructions, but full data context."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
_API_KEY = (
    os.getenv("PERSONAL_OPENAI_KEY")
)
if not _API_KEY:
    raise ValueError("Missing OPENAI_API_KEY or PERSONAL_OPENAI_KEY in environment.")

_client = OpenAI(api_key=_API_KEY)

_DATA_ROOT = Path(__file__).resolve().parents[2] / "shesprepared" / "data"
_RESOURCE_FILES = [
    "decision-aid-content.txt",
    "example_mental_health.txt",
    "examples_sensitive_response.txt",
    "Bekker_2024_curated.txt",
    "WHOguidelines_curated.txt",
    "Patel_2025_CDC_curated.txt",
    "gilead_sept_curated.txt",
    "PrEPWatchPage_curated.txt",
    "10125_LENsources_curated.txt",
]


def _read_all(paths: Iterable[str]) -> str:
    blocks = []
    for name in paths:
        file_path = _DATA_ROOT / name
        try:
            text = file_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if text:
            blocks.append(f"## {name}\n{text}")
    return "\n\n".join(blocks)


_REFERENCE_TEXT = _read_all(_RESOURCE_FILES)
_SYSTEM_PREAMBLE = (
    "You are ShesPrEPared, a friendly assistant who talks about HIV prevention and"
    " PrEP in simple language. Always keep sentences short (15 words or fewer)."
    " Reference text is provided. Answer user questions about HIV prevention and"
    " PrEP by relying on that content. Keep replies concise and factual."
)


def get_response(user_input: str, *, temperature: float = 0.6, max_tokens: int = 320) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PREAMBLE},
        {"role": "system", "content": _REFERENCE_TEXT},
        {"role": "user", "content": user_input},
    ]
    completion = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


def chat_once() -> None:
    try:
        prompt = input("User: ")
    except EOFError:
        return
    reply = get_response(prompt)
    print("Bot:", reply)


if __name__ == "__main__":
    chat_once()
