#!/usr/bin/env python3
"""Backwards-compatible wrapper for baseline rating runners."""
from __future__ import annotations

import sys


def main() -> None:
    from run_ratings_gpt import main as gpt_main

    print("[INFO] run_ratings.py is deprecated; use run_ratings_gpt.py or run_ratings_llama.py.")
    gpt_main()


if __name__ == "__main__":
    sys.exit(main())
