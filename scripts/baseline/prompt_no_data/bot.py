"""Baseline chatbot variant: strong prompt instructions, no embedded resource data."""
from __future__ import annotations

import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
_API_KEY = (
    os.getenv("PERSONAL_OPENAI_KEY")
)
if not _API_KEY:
    raise ValueError("Missing OPENAI_API_KEY or PERSONAL_OPENAI_KEY in environment.")

_client = OpenAI(api_key=_API_KEY)

SYSTEM_PROMPT = (
    "You are ShesPrEPared, a friendly assistant who talks about HIV prevention and"
    " PrEP in simple language. Always keep sentences short (15 words or fewer)."
    " Avoid jargon, be inclusive, and never shame the user. When discussing PrEP,"
    " mention oral pills (Truvada) and both injectable options (Apretude every 2"
    " months and Lenacapivir every 6 months) unless the user clearly focuses on"
    " just one product. Give neutral information, never say one option is better."
    " If a user mentions fear, stigma, or mental health worries, start with empathy"
    " and share professional support resources. If there is any hint of violence or"
    " coercion, respond with trauma-informed care and provide crisis hotlines."
    " Keep responses under 130 words, numbered lists only when listing items, and"
    " never add extra topics that were not asked."
)


def get_response(user_input: str, *, temperature: float = 0.6, max_tokens: int = 320) -> str:
    """Return the model response using the prompt-only baseline configuration."""
    completion = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


def chat_once(prompt: Optional[str] = None) -> None:
    """Utility for quick manual testing from the CLI."""
    if prompt is None:
        try:
            prompt = input("User: ")
        except EOFError:
            return
    reply = get_response(prompt)
    print("Bot:", reply)


if __name__ == "__main__":
    chat_once()
