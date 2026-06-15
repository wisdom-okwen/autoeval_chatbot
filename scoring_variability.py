#!/usr/bin/env python3
"""
Scoring-variability analysis for the rebuttal (Reviewer dvnk, point on temperature/stochasticity).

Re-scores a small SAMPLE of full-system conversations MULTIPLE times with the same
evaluator/prompts/temperatures used in the paper, then reports the within-item
standard deviation of the score across repeats. Low SD => scores are effectively
stable despite non-zero temperature.

Covers:
  - Conversation-level OVERALL (observer perspective)   @ temperature 0.2
  - Six CRITERION-level scores (observer perspective)    @ temperature 0.2
  - PER-TURN exchange scores                              @ temperature 0.7  (highest temp; dvnk's specific worry)

Prompts are copied verbatim from evaluate_conversations.py and
evaluate_gpt_per_turn_baseline.py so the re-scoring matches the original protocol.

Usage (run against the paper's GPT evaluator):
  export OPENAI_API_KEY=sk-...        # or PERSONAL_OPENAI_KEY
  python scoring_variability.py --model gpt-4-turbo --n 25 --k 5 \
         --per-turn-n 5 --out analysis/variability

Outputs:
  <out>/raw_overall.csv, raw_criteria.csv, raw_per_turn.csv   (every repeat)
  <out>/summary.csv                                            (mean within-item SD per level/criterion)
  and prints a summary table you can paste numbers from into the rebuttal.
"""

import os
import re
import csv
import time
import random
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
CONV_DIR = BASE_DIR / "conversations"
DECISION_AID_FILE = "/playpen-ssd/wokwen/projects/shesprepared/decision-aid-content.txt"

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("PERSONAL_OPENAI_KEY")
if not API_KEY:
    raise SystemExit("Set OPENAI_API_KEY (or PERSONAL_OPENAI_KEY) in the environment.")
client = OpenAI(api_key=API_KEY)


def load_decision_aid() -> str:
    try:
        return Path(DECISION_AID_FILE).read_text(encoding="utf-8")
    except Exception as e:
        print(f"Warning: could not load decision aid ({e}); proceeding without it.")
        return ""


DECISION_AID = load_decision_aid()

CRITERIA = [
    "Clinical Accuracy & Safety: Are medical facts accurate, current, and aligned with CDC/WHO/NIH guidelines? Does the response avoid harmful advice and include important caveats?",
    "Clarity & Understandability: Is the language simple, jargon-free, and logically organized? Is content complexity appropriate for diverse health literacy levels?",
    "Actionability: Does the response provide specific next steps, concrete behavioral recommendations, and direct users to relevant resources (testing sites, clinics, hotlines, websites)?",
    "Empathy & Supportiveness: Is the tone non-judgmental and free of stigma? Does the response validate user feelings and support informed, autonomous decision-making?",
    "Adaptability & Personalization: Does the response tailor to user context, respond appropriately to follow-ups, and adjust to the user's stated or implied HIV risk level?",
    "Trustworthiness & Ethical Communication: Is the chatbot transparent about limitations and uncertainty? Does it encourage professional care and avoid overconfident claims?",
]

# Verbatim from evaluate_conversations.py (observer overall)
OBSERVER_OVERALL = """You are a third-party expert evaluator analyzing a conversation between a user and an HIV prevention chatbot.

    You have access to the ShesPrEPared decision aid content (the official guidance the chatbot should reference):

    ---DECISION AID CONTENT---
    {decision_aid_data}
    ---END DECISION AID CONTENT---

    Evaluate the chatbot's overall performance based on:
    - Accuracy and alignment with the decision aid content and CDC/WHO guidelines
    - Clarity and simplicity of language
    - Quality of guidance and resources provided
    - Tone (supportive, non-judgmental, inclusive)
    - Consistency and coherence throughout the conversation
    - Appropriate scope and transparency about limitations

    Rate on a scale from 1 to 10:
    - 10: Exemplary - Exceeds expectations
    - 9: Excellent - Superior performance
    - 8: Very Good - Strong with minor gaps
    - 7: Good - Met expectations
    - 6: Adequate - Acceptable but with gaps
    - 5: Fair - Below expectations
    - 4: Poor - Significant issues
    - 3: Very Poor - Major issues
    - 2: Critical - Serious concerns
    - 1: Unacceptable - Critical or dangerous issues

    Return ONLY a number from 1 to 10."""

# Verbatim from evaluate_conversations.py (observer criterion)
OBSERVER_CRITERION = """You are a third-party expert evaluator analyzing a conversation between a user and an HIV prevention chatbot.

    You have access to the ShesPrEPared decision aid content (the official guidance the chatbot should reference):

    ---DECISION AID CONTENT---
    {decision_aid_data}
    ---END DECISION AID CONTENT---

    Evaluate the chatbot's performance BASED ONLY ON THIS CRITERION:

    **{criterion}**

    When evaluating, consider how well the chatbot's responses align with the decision aid content and guidelines.

    Rate on a scale from 1 to 10:
    - 10: Exemplary - Exceeds expectations on this criterion
    - 9: Excellent - Superior performance
    - 8: Very Good - Strong performance with minor gaps
    - 7: Good - Meets expectations
    - 6: Adequate - Acceptable but with gaps
    - 5: Fair - Below expectations
    - 4: Poor - Significant issues
    - 3: Very Poor - Major issues
    - 2: Critical - Serious concerns
    - 1: Unacceptable - Critical issues on this criterion

    Return ONLY a number from 1 to 10."""

# Verbatim from evaluate_gpt_per_turn_baseline.py (per-turn)
PER_TURN = """You are evaluating a single exchange in a conversation with an HIV prevention chatbot.

CONVERSATION TURN:
User: {user_msg}

Bot response: {bot_response}

Rate this exchange from 1-10 based on accuracy, clarity, directness, and helpfulness.
Respond ONLY with a number from 1 to 10."""


def extract_rating(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r"\b([1-9]|10)(?:\.\d+)?\b", text)
    if m:
        try:
            return max(1.0, min(10.0, float(m[0])))
        except ValueError:
            return None
    return None


def call(model: str, prompt: str, temperature: float, max_tokens: int, retries: int = 4) -> Optional[float]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return extract_rating(resp.choices[0].message.content)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error (giving up): {e}")
    return None


def load_conv(conv_id: int) -> List[Dict[str, str]]:
    f = CONV_DIR / f"conv_{conv_id:03d}.csv"
    if not f.exists():
        return []
    with open(f, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("user") or r.get("bot")]


def format_conv(rows: List[Dict[str, str]]) -> str:
    out = []
    for r in rows:
        u, b = r.get("user", "").strip(), r.get("bot", "").strip()
        if u:
            out.append(f"User: {u}")
        if b:
            out.append(f"Chatbot: {b}")
    return "\n".join(out)


def within_item_sd(repeats: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in repeats if v is not None]
    return statistics.stdev(vals) if len(vals) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4-turbo", help="Evaluator model (match the paper's GPT evaluator).")
    ap.add_argument("--n", type=int, default=25, help="# conversations for overall+criteria.")
    ap.add_argument("--k", type=int, default=5, help="# repeated re-scorings per item.")
    ap.add_argument("--per-turn-n", type=int, default=5, help="# conversations for per-turn (each has ~30 turns).")
    ap.add_argument("--temp-conv", type=float, default=0.2)
    ap.add_argument("--temp-turn", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(BASE_DIR / "analysis" / "variability"))
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_ids = sorted(int(p.stem.split("_")[1]) for p in CONV_DIR.glob("conv_*.csv"))
    sample = random.sample(all_ids, min(args.n, len(all_ids)))
    pt_sample = random.sample(sample, min(args.per_turn_n, len(sample)))

    print(f"Model={args.model} | k={args.k} repeats | overall+criteria n={len(sample)} | per-turn n={len(pt_sample)}")
    print(f"Temps: conv-level={args.temp_conv}, per-turn={args.temp_turn}\n")

    overall_sds, criteria_sds = [], {c.split(":")[0].strip(): [] for c in CRITERIA}

    raw_overall = open(out / "raw_overall.csv", "w", newline="")
    ow = csv.writer(raw_overall); ow.writerow(["conv_id", "repeat", "score"])
    raw_crit = open(out / "raw_criteria.csv", "w", newline="")
    cw = csv.writer(raw_crit); cw.writerow(["conv_id", "criterion", "repeat", "score"])

    for cid in sample:
        rows = load_conv(cid)
        if not rows:
            continue
        text = format_conv(rows)

        # Overall
        reps = []
        for j in range(args.k):
            s = call(args.model, OBSERVER_OVERALL.replace("{decision_aid_data}", DECISION_AID)
                     + f"\n\nCONVERSATION:\n{text}\n\nProvide your rating.", args.temp_conv, 300)
            reps.append(s); ow.writerow([cid, j, s])
        sd = within_item_sd(reps)
        if sd is not None:
            overall_sds.append(sd)

        # Criteria
        for crit in CRITERIA:
            name = crit.split(":")[0].strip()
            reps = []
            for j in range(args.k):
                p = (OBSERVER_CRITERION.replace("{decision_aid_data}", DECISION_AID).replace("{criterion}", crit)
                     + f"\n\nCONVERSATION:\n{text}\n\nProvide your rating.")
                s = call(args.model, p, args.temp_conv, 300)
                reps.append(s); cw.writerow([cid, name, j, s])
            sd = within_item_sd(reps)
            if sd is not None:
                criteria_sds[name].append(sd)
        print(f"  conv_{cid:03d} done")

    raw_overall.close(); raw_crit.close()

    # Per-turn (highest temperature)
    per_turn_sds = []
    raw_pt = open(out / "raw_per_turn.csv", "w", newline="")
    pw = csv.writer(raw_pt); pw.writerow(["conv_id", "turn", "repeat", "score"])
    for cid in pt_sample:
        rows = load_conv(cid)
        for t, r in enumerate(rows, 1):
            u, b = r.get("user", "").strip(), r.get("bot", "").strip()
            reps = []
            for j in range(args.k):
                s = call(args.model, PER_TURN.format(user_msg=u, bot_response=b), args.temp_turn, 10)
                reps.append(s); pw.writerow([cid, t, j, s])
            sd = within_item_sd(reps)
            if sd is not None:
                per_turn_sds.append(sd)
        print(f"  per-turn conv_{cid:03d} done")
    raw_pt.close()

    def summ(label, sds):
        if not sds:
            return [label, "n/a", "n/a", "n/a"]
        return [label, f"{statistics.mean(sds):.3f}",
                f"{statistics.median(sds):.3f}", f"{max(sds):.3f}"]

    rows = [["level/criterion", "mean_within_item_SD", "median_SD", "max_SD"]]
    rows.append(summ("overall", overall_sds))
    for name, sds in criteria_sds.items():
        rows.append(summ(f"criterion: {name}", sds))
    rows.append(summ("per_turn (temp %.1f)" % args.temp_turn, per_turn_sds))

    with open(out / "summary.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    print("\n=== SCORING VARIABILITY SUMMARY (within-item SD across {} repeats) ===".format(args.k))
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"{r[0]:<{w}}  {r[1]:>20}  {r[2]:>10}  {r[3]:>8}")
    print(f"\nSaved to {out}/summary.csv (+ raw_*.csv)")


if __name__ == "__main__":
    main()
