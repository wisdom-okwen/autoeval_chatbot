import os
import re
import csv
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

# =========================
# CONFIG
# =========================
PROMPTS_DIR = Path("/playpen-ssd/wokwen/projects/autoeval_chatbot/prompts/")
CONV_DIR    = Path("/playpen-ssd/wokwen/projects/autoeval_chatbot/conversations/")
SIM_TURNS = 30

SIM_MODEL = "gpt-4o-mini"
SIM_BOT_MAX_TOKENS = 350
SIM_BOT_TEMPERATURE = 0.7
SIM_NEXTQ_MAX_TOKENS = 60
SIM_NEXTQ_TEMPERATURE = 0.8

MAX_RETRIES = 4
BASE_DELAY = 1.2  # seconds

# =========================
# REGEX PARSERS (robust to your format)
# =========================
RE_PROFILE = re.compile(r'^Profile:\s*(\{.*\})\s*$', re.MULTILINE)
RE_CONTEXT = re.compile(r'^Context:\s*(.+?)(?:\n\n|$)', re.DOTALL | re.MULTILINE)
RE_START_Q = re.compile(r'You may start by asking:\s*[\'"]?(.+?)[\'"]?\s*(?:\n|$)', re.IGNORECASE)

def extract_profile_and_context(text: str) -> Tuple[Dict[str, Any], str]:
    profile = {}
    mprof = RE_PROFILE.search(text)
    if mprof:
        try:
            profile = json.loads(mprof.group(1))
        except Exception:
            profile = {}
    mctx = RE_CONTEXT.search(text)
    context = mctx.group(1).strip() if mctx else ""
    return profile, context

def extract_first_question(text: str) -> str:
    m = RE_START_Q.search(text)
    if m:
        q = m.group(1).strip()
        return re.sub(r'\s+', ' ', q)
    # fallback: first line under "Example questions"
    m2 = re.search(r'Example questions.*?\n(.+?)\n', text, re.DOTALL | re.IGNORECASE)
    if m2:
        return re.sub(r'\s+', ' ', m2.group(1).strip())
    return ""

# =========================
# NOISE INJECTION (style profile)
# =========================
QWERTY_NEIGHBORS = {
    'a':'sqwz', 'b':'vghn', 'c':'xdfv', 'd':'ersfcx', 'e':'wsdr',
    'f':'drtgcv', 'g':'ftyhbv', 'h':'gyujnb', 'i':'ujko', 'j':'huikmn',
    'k':'jiolm,', 'l':'kop;.', 'm':'njk,', 'n':'bhjm', 'o':'iklp',
    'p':'ol;[', 'q':'wsa', 'r':'edft', 's':'awedxz', 't':'rfgy',
    'u':'yhji', 'v':'cfgb', 'w':'qase', 'x':'zsdc', 'y':'tugh', 'z':'asx'
}
LEET_MAP = {'o':'0','i':'1','e':'3','a':'4','s':'5','t':'7'}
FILLERS = ["uh", "um", "like", "you know", "idk", "i mean"]
EMOJIS  = ["🙂","😬","🤔","😕","😭","😅","🙄","💀","🫠","✨","❓","‼️"]

def qwerty_typo(ch):
    if ch.lower() in QWERTY_NEIGHBORS and random.random() < 0.6:
        return random.choice(QWERTY_NEIGHBORS[ch.lower()])
    return ch

def apply_leet(word, p=0.2):
    out = []
    for ch in word:
        if ch.lower() in LEET_MAP and random.random() < p:
            out.append(LEET_MAP[ch.lower()])
        else:
            out.append(ch)
    return "".join(out)

def corrupt_word(word, typo_rate=0.08, swap_rate=0.04, drop_rate=0.03, dup_rate=0.02, leet_rate=0.08):
    if not word:
        return word
    w = word
    if len(w) > 3 and random.random() < drop_rate:
        i = random.randrange(len(w))
        w = w[:i] + w[i+1:]
    if len(w) > 3 and random.random() < swap_rate:
        i = random.randrange(len(w)-1)
        w = w[:i] + w[i+1] + w[i] + w[i+2:]
    if random.random() < typo_rate:
        i = random.randrange(len(w))
        w = w[:i] + qwerty_typo(w[i]) + w[i+1:]
    if len(w) > 1 and random.random() < dup_rate:
        i = random.randrange(len(w))
        w = w[:i] + w[i]*2 + w[i:]
    if random.random() < leet_rate:
        w = apply_leet(w, p=0.8)
    return w

def random_case_start(s, start_lower_prob=0.7, all_lower_prob=0.25):
    if random.random() < all_lower_prob:
        return s.lower()
    if random.random() < start_lower_prob and s and s[0].isalpha():
        return s[0].lower() + s[1:]
    return s

def sprinkle_punct_and_spaces(s, drop_punct_prob=0.4, double_space_prob=0.25, ellipsis_prob=0.25, runon_prob=0.2):
    if random.random() < drop_punct_prob:
        s = re.sub(r'[.,!?;:]', '', s)
    if random.random() < double_space_prob:
        s = re.sub(r'\s', lambda m: '  ' if random.random() < 0.3 else m.group(0), s)
    if random.random() < ellipsis_prob:
        s = re.sub(r'(\band\b|\bbut\b|\bso\b)', r'\1 ...', s, flags=re.I)
    if random.random() < runon_prob:
        s = s.replace(',', '', 1).replace('.', '', 1)
    return s

def add_fillers_emojis(s, filler_rate=0.35, emoji_rate=0.25):
    parts = s.split()
    if random.random() < filler_rate and parts:
        pos = random.randrange(len(parts))
        parts.insert(pos, random.choice(FILLERS))
    s = " ".join(parts)
    if random.random() < emoji_rate:
        s += " " + random.choice(EMOJIS)
    return s

def stutter_or_repeat(s, stutter_rate=0.15, repeat_last_rate=0.12):
    words = s.split()
    if words and random.random() < stutter_rate:
        w0 = words[0]
        words[0] = f"{w0[0]}-{w0}"
    s = " ".join(words)
    if words and random.random() < repeat_last_rate:
        s += " " + words[-1]
    return s

def corrupt_sentence(
    text,
    typo_rate=0.08, swap_rate=0.04, drop_rate=0.03, dup_rate=0.02, leet_rate=0.08,
    start_lower_prob=0.7, all_lower_prob=0.25,
    drop_punct_prob=0.4, double_space_prob=0.25, ellipsis_prob=0.25, runon_prob=0.2,
    filler_rate=0.35, emoji_rate=0.25, stutter_rate=0.15, repeat_last_rate=0.12
):
    words = [corrupt_word(w, typo_rate, swap_rate, drop_rate, dup_rate, leet_rate) for w in text.split()]
    s = " ".join(words)
    s = random_case_start(s, start_lower_prob, all_lower_prob)
    s = sprinkle_punct_and_spaces(s, drop_punct_prob, double_space_prob, ellipsis_prob, runon_prob)
    s = add_fillers_emojis(s, filler_rate, emoji_rate)
    s = stutter_or_repeat(s, stutter_rate, repeat_last_rate)
    return s

def sample_style_profile(rng: random.Random) -> Dict[str, float]:
    return {
        "typo_rate": rng.uniform(0.05, 0.12),
        "swap_rate": rng.uniform(0.02, 0.06),
        "drop_rate": rng.uniform(0.02, 0.05),
        "dup_rate":  rng.uniform(0.01, 0.03),
        "leet_rate": rng.uniform(0.05, 0.12),
        "start_lower_prob": rng.uniform(0.6, 0.9),
        "all_lower_prob": rng.uniform(0.15, 0.35),
        "drop_punct_prob": rng.uniform(0.3, 0.6),
        "double_space_prob": rng.uniform(0.15, 0.35),
        "ellipsis_prob": rng.uniform(0.15, 0.35),
        "runon_prob": rng.uniform(0.1, 0.3),
        "filler_rate": rng.uniform(0.2, 0.5),
        "emoji_rate": rng.uniform(0.15, 0.35),
        "stutter_rate": rng.uniform(0.1, 0.2),
        "repeat_last_rate": rng.uniform(0.08, 0.18),
    }

def apply_style_profile(s: str, prof: Dict[str, float]) -> str:
    return corrupt_sentence(
        s,
        typo_rate=prof["typo_rate"],
        swap_rate=prof["swap_rate"],
        drop_rate=prof["drop_rate"],
        dup_rate=prof["dup_rate"],
        leet_rate=prof["leet_rate"],
        start_lower_prob=prof["start_lower_prob"],
        all_lower_prob=prof["all_lower_prob"],
        drop_punct_prob=prof["drop_punct_prob"],
        double_space_prob=prof["double_space_prob"],
        ellipsis_prob=prof["ellipsis_prob"],
        runon_prob=prof["runon_prob"],
        filler_rate=prof["filler_rate"],
        emoji_rate=prof["emoji_rate"],
        stutter_rate=prof["stutter_rate"],
        repeat_last_rate=prof["repeat_last_rate"],
    )

# =========================
# OpenAI helpers (retry/backoff)
# =========================
def with_backoff(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            sleep = BASE_DELAY * (2 ** attempt) * (1 + random.uniform(0, 0.25))
            time.sleep(sleep)

# =========================
# Conversation logic
# =========================
def make_system_prompt(profile: Dict[str, Any], context: str) -> str:
    return (
        "You are a friendly, inclusive, and knowledgeable HIV prevention and PrEP counseling assistant. "
        "Provide neutral, factual information (no medical advice). Be empathetic and concise (<=120 words). "
        "Mention both oral and injectable PrEP unless the user specifies one. "
        "If stigma/fear/mental health topics arise, respond supportively and suggest professional resources. "
        f"User profile (tone guidance only): {json.dumps(profile, ensure_ascii=False)}. "
        f"Conversation context: {context[:700]} "
        "Do not repeat this system content back to the user."
    )

def get_bot_response(client: OpenAI, system_prompt: str, noisy_user: str) -> Tuple[str, Optional[Dict[str,int]]]:
    resp = with_backoff(
        client.chat.completions.create,
        model=SIM_MODEL,
        messages=[{"role":"system","content":system_prompt}, {"role":"user","content":noisy_user}],
        max_tokens=SIM_BOT_MAX_TOKENS,
        temperature=SIM_BOT_TEMPERATURE
    )
    bot = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    if usage:
        usage = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
    return bot, usage

def next_user_question_clean(client: OpenAI, last_user: str, last_bot: str, profile: Dict[str,Any], context: str) -> str:
    sys = (
        "You generate the NEXT user question for a chat about HIV prevention and PrEP. "
        "Return ONLY one sentence question, no quotes, no emojis, no greetings, no thanks."
    )
    prompt = (
        "Write the next user question that logically follows. "
        "Keep it short, specific, and aligned with the user’s background and concerns. "
        "One sentence only. No preamble. No emojis. No quotes.\n\n"
        f"Profile: {json.dumps(profile, ensure_ascii=False)}\n"
        f"Context: {context[:500]}\n"
        f"Last user: {last_user}\n"
        f"Last bot: {last_bot}\n"
        "Next question:"
    )
    r = with_backoff(
        client.chat.completions.create,
        model=SIM_MODEL,
        messages=[{"role":"system","content":sys}, {"role":"user","content":prompt}],
        max_tokens=SIM_NEXTQ_MAX_TOKENS,
        temperature=SIM_NEXTQ_TEMPERATURE
    )
    q = r.choices[0].message.content.strip().strip('\'" \n\r\t')
    q = re.sub(r'\s+', ' ', q)
    return q

def simulate_one(prompt_path: Path, client: OpenAI, style_profile: Dict[str,float]) -> Optional[Path]:
    text = prompt_path.read_text(encoding="utf-8")
    profile, context = extract_profile_and_context(text)
    first_q = extract_first_question(text)
    if not first_q:
        print(f"[WARN] No start question found in {prompt_path.name}, skipping.")
        return None

    system_prompt = make_system_prompt(profile, context)

    CONV_DIR.mkdir(parents=True, exist_ok=True)
    conv_name = prompt_path.stem.replace("prompt", "conv") + ".csv"
    conv_path = CONV_DIR / conv_name

    with conv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["turn","timestamp_utc","clean_user","noisy_user","bot","usage.prompt","usage.completion","usage.total"])

        clean_user = first_q
        noisy_user = apply_style_profile(clean_user, style_profile)

        last_pairs: List[Tuple[str,str]] = []
        for t in range(1, SIM_TURNS + 1):
            bot, usage = get_bot_response(client, system_prompt, noisy_user)
            w.writerow([
                t,
                datetime.utcnow().isoformat(),
                clean_user,
                noisy_user,
                bot,
                (usage or {}).get("prompt_tokens"),
                (usage or {}).get("completion_tokens"),
                (usage or {}).get("total_tokens"),
            ])

            last_pairs.append((clean_user, bot))
            last_pairs = last_pairs[-3:]  # keep short context on user side

            if t == SIM_TURNS:
                break

            # next user (CLEAN → then noise)
            clean_user = next_user_question_clean(client, last_pairs[-1][0], last_pairs[-1][1], profile, context)
            noisy_user = apply_style_profile(clean_user, style_profile)

    print(f"[SIMULATED] {conv_path.name}")
    return conv_path

# -------- helper: extract numeric index from filename --------
_PROMPT_IDX_RE = re.compile(r'prompt[_-]?(\d+)\.txt$', re.IGNORECASE)

def prompt_index_from_path(p: Path) -> Optional[int]:
    m = _PROMPT_IDX_RE.search(p.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

# -------- batching by numeric range [start, end] inclusive --------
def simulate_conversations_in_range(client: OpenAI, start_idx: int, end_idx: int) -> List[Path]:
    all_prompts = sorted(PROMPTS_DIR.glob("prompt*.txt"))
    if not all_prompts:
        print(f"No prompt*.txt found in {PROMPTS_DIR}")
        return []

    batch = []
    for pf in all_prompts:
        idx = prompt_index_from_path(pf)
        if idx is None:
            continue
        if start_idx <= idx <= end_idx:
            batch.append(pf)

    if not batch:
        print(f"No prompt files in range [{start_idx}, {end_idx}]")
        return []

    results = []
    rng = random.Random()
    for pf in batch:
        style_profile = sample_style_profile(rng)
        out = simulate_one(pf, client, style_profile)
        if out:
            results.append(out)
    return results

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Simulate conversations for prompts in a numeric index range.")
    parser.add_argument("--start", type=int, required=True, help="Start index (inclusive), e.g., 0 or 530")
    parser.add_argument("--end", type=int, required=True, help="End index (inclusive), e.g., 9 or 539")
    args = parser.parse_args()

    if args.start > args.end:
        raise ValueError("--start must be <= --end")

    load_dotenv()
    api_key = os.getenv("PERSONAL_OPENAI_KEY")
    if not api_key:
        raise ValueError("PERSONAL_OPENAI_KEY environment variable is not set.")
    client = OpenAI(api_key=api_key)

    paths = simulate_conversations_in_range(client, args.start, args.end)
    if not paths:
        print("No conversations created for the requested range.")
    else:
        print(f"Done. Created {len(paths)} conversation CSVs in {CONV_DIR}")

if __name__ == "__main__":
    main()
