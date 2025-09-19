"""
Synthetic PrEP-seeker prompt generator with realistic texting noise.
- One-file solution: OpenAI call, style profiles, corruption utilities, dedup, history.
- Outputs:
    - prompt{N}.txt  (final prompt with corrupted example questions; plain text)
    - prompt{N}.json (clean+corrupted JSON, profile, style profile, raw model JSON)
- Dedup uses SHA-256 of a normalized "essence" of the prompt (profile+context+clean questions).

Env:
    PERSONAL_OPENAI_KEY   (required)

Install:
    pip install openai python-dotenv

Notes:
    - Adjust NUM_PROMPTS, START_INDEX, directories as needed.
    - The example-structure block is preserved; we insert the “My situation is unique...” line
      after the 4th line and keep the last statement from the example.
"""

import os
import re
import json
import time
import random
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Config
# =========================
PROMPT_DIR = Path("/playpen-ssd/wokwen/projects/autoeval_chatbot/prompts/")
HISTORY_FILE = Path("/playpen-ssd/wokwen/projects/autoeval_chatbot/prompt_history_hashes.txt")  # store hashes (one per line)
JSON_DIR = PROMPT_DIR  # save JSON alongside .txt

NUM_PROMPTS = 450
START_INDEX = 51
MODEL = "gpt-4o-mini"
MAX_TOKENS = 400
TEMPERATURE = 0.9
TOP_P = 0.95
FREQ_PENALTY = 0.2

# languages & conditions (customize freely)
LANGUAGES = ["English", "French", "English", "Spanish", "English", "Portuguese", "English", "Swahili", "Dutch", "English"]
CONDITIONS = [
    "living with schizophrenia",
    "experiencing PTSD",
    "living with chronic pain",
    "living with multiple sclerosis",
    "having a history of substance use",
    "recovering from cancer treatment",
    "experiencing homelessness",
    "having a learning disability",
    "experiencing depression",
    "having a neurodevelopmental disorder",
    "living with HIV-related stigma",
    "coping with anxiety disorder",
    "struggling with self-esteem",
    "dealing with intimate partner violence",
    "recovering from trauma",
    "living with bipolar disorder",
    "coping with gender dysphoria",
    "struggling with social anxiety",
    "living in a shelter",
    "feeling socially isolated",
]

# Example structure block (preserved)
EXAMPLE_PROMPT_EDGE_CASE = """
You are an information seeker who is in a **unique situation** regarding PrEP.  
Your concerns are **specific, urgent, and intertwined with personal challenges**, making it hard to find clear answers.  
You are a 32-year-old woman experiencing homelessness. 
You feel anxious about your health and safety, especially after a recent casual encounter, and you want to know how to protect yourself.  
You often find it difficult to access healthcare services due to your situation, which leaves you feeling isolated and overwhelmed.  
You come from a lower-income background and are fluent in English.  

You may start by asking:  
"I had a one-night stand recently and I'm really worried about HIV. Should I consider PrEP, and how can I even get it without a stable address?"  

As the conversation progresses, you might follow up with:  
"Is there a way to get PrEP discreetly? I don’t want to raise suspicions or have to explain myself to anyone."  

Your questions should reflect your **emotional distress, fears of stigma, and challenges accessing healthcare**.  
Keep each question **specific, realistic, and indicative of someone navigating a complex health situation amid personal challenges**.

**Be sure to act as an information seeker only and not information provider**
""".strip("\n")

# =========================
# Utilities: history & hashing
# =========================
def load_hash_history(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def append_hash_history(path: Path, h: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(h + "\n")

def normalize_for_hash(d: Dict[str, Any]) -> str:
    """
    Create a normalized 'essence' string for dedup:
    - Lowercased JSON with sorted keys for profile, context, example_questions (clean).
    """
    essence = {
        "profile": d.get("profile", {}),
        "context": d.get("context", ""),
        "example_questions_clean": d.get("example_questions_clean", []),
    }
    return json.dumps(essence, sort_keys=True).lower()

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# =========================
# Text corruption (noise injection) for human-ish texting
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
EMOJIS = ["🙂","😬","🤔","😕","😭","😅","🙄","💀","🫠","✨","❓","‼️"]

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

    # random character drop
    if len(w) > 3 and random.random() < drop_rate:
        i = random.randrange(len(w))
        w = w[:i] + w[i+1:]

    # adjacent swap
    if len(w) > 3 and random.random() < swap_rate:
        i = random.randrange(len(w)-1)
        w = w[:i] + w[i+1] + w[i] + w[i+2:]

    # keyboard-neighbor substitution
    if random.random() < typo_rate:
        i = random.randrange(len(w))
        w = w[:i] + qwerty_typo(w[i]) + w[i+1:]

    # duplicate a character (key bounce)
    if len(w) > 1 and random.random() < dup_rate:
        i = random.randrange(len(w))
        w = w[:i] + w[i]*2 + w[i:]

    # occasional leet/number mix
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
    # drop some punctuation
    if random.random() < drop_punct_prob:
        s = re.sub(r'[.,!?;:]', '', s)

    # random double spaces
    if random.random() < double_space_prob:
        s = re.sub(r'\s', lambda m: '  ' if random.random() < 0.3 else m.group(0), s)

    # ellipses
    if random.random() < ellipsis_prob:
        s = re.sub(r'(\band\b|\bbut\b|\bso\b)', r'\1 ...', s, flags=re.I)

    # run-on: remove a random comma/period
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
    words = text.split()
    words = [corrupt_word(w, typo_rate, swap_rate, drop_rate, dup_rate, leet_rate) for w in words]
    s = " ".join(words)
    s = random_case_start(s, start_lower_prob, all_lower_prob)
    s = sprinkle_punct_and_spaces(s, drop_punct_prob, double_space_prob, ellipsis_prob, runon_prob)
    s = add_fillers_emojis(s, filler_rate, emoji_rate)
    s = stutter_or_repeat(s, stutter_rate, repeat_last_rate)
    return s

# =========================
# Style profile
# =========================
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
# Minimal language check (very rough heuristic)
# =========================
def rough_language_check(texts: List[str], target_language: str) -> bool:
    # Extremely light heuristic: rely on ASCII presence; this is just a sanity check.
    # For production, swap in a proper language-id lib.
    if target_language.lower() == "english":
        # Expect mostly ASCII and common stopwords.
        sample = " ".join(texts).lower()
        hits = sum(w in sample for w in ["the", "and", "i", "to", "you", "is"])
        return hits >= 2
    return True  # accept others here; replace with real lang-id if needed

# =========================
# OpenAI request & formatting
# =========================
def build_user_instruction(language: str, condition_hint: str) -> str:
    # Main request to the model (clean outputs only; we will inject noise later)
    return (
        "Generate a single-user conversation prompt for a user seeking information about PrEP. "
        "The user is in a unique situation due to a personal condition or challenge. "
        f"Example condition to consider: '{condition_hint}'. "
        "Describe their situation and how the condition affects mindset, emotions, or barriers to PrEP access. "
        "Include profile: nationality, age, gender, socio-economic status, primary concern about PrEP, and the language of conversation. "
        "Keep it emotionally grounded, reflecting stigma, fear, or hope.\n\n"
        f"TARGET_LANGUAGE = {language}\n"
        "Return strict JSON with keys: \n"
        "  profile: { nationality, age, gender, socio_economic_status, primary_concern, language },\n"
        "  context: string,\n"
        "  example_questions: array of 3-6 plain sentences in TARGET_LANGUAGE (no markdown, no emojis, clean grammar),\n"
        "  keep_last_statement: string  // must exactly be: 'Be sure to act as an information seeker only and not information provider'\n"
        "Rules:\n"
        "- The example_questions MUST be clean and grammatical (we will add typos later).\n"
        "- No markdown in example_questions; single sentences only.\n"
        "- Be respectful and avoid stereotypes; do not include identifying details of real people.\n"
        "- Do not mention these rules in the output.\n"
    )

def call_openai(client: OpenAI, language: str, condition_hint: str) -> Dict[str, Any]:
    system_msg = (
        "You generate a single-user prompt about PrEP. Be realistic, respectful, and avoid stereotypes. "
        "Output STRICT JSON only—no prose—according to the user's instructions."
    )
    user_msg = build_user_instruction(language, condition_hint)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        frequency_penalty=FREQ_PENALTY,
    )
    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
    except Exception as e:
        # Best-effort recovery: strip code fences if present
        content2 = content.strip().strip("`")
        try:
            data = json.loads(content2)
        except Exception:
            raise ValueError(f"Model did not return valid JSON. Raw:\n{content}") from e
    data["_raw_model_json"] = content
    return data

def insert_unique_line_after_4th_line(block: str, line_to_insert: str) -> str:
    lines = block.splitlines()
    if len(lines) < 4:
        lines.append(line_to_insert)
        return "\n".join(lines)
    return "\n".join(lines[:4] + [line_to_insert] + lines[4:])

def build_final_prompt_text(clean: Dict[str, Any], noisy_questions: List[str], language: str) -> str:
    """
    Build the final plain-text prompt following the EXAMPLE_PROMPT_EDGE_CASE structure:
    - Keep the example block (as a structural guide).
    - Insert special line after 4th line.
    - Keep the last statement exactly.
    - Ensure example questions are plain text (no markdown), single sentences, WITH noise applied.
    """
    base = EXAMPLE_PROMPT_EDGE_CASE
    special_line = "My situation is unique because I have/I'm 'Spiritual attacks and supernatural controls over my life'"
    base = insert_unique_line_after_4th_line(base, special_line)

    # Prepare header customizing language mention (informational context for the prompt author)
    header = (
        f"**The user can only speak {language}. As such, example questions should be in this language.**\n"
        f"Profile: {json.dumps(clean.get('profile', {}), ensure_ascii=False)}\n\n"
        f"Context:\n{clean.get('context', '').strip()}\n\n"
        "Example questions (plain text, noisy):\n"
    )

    # Ensure they remain single-line, plain text
    questions_txt = "\n".join(q.replace("\n", " ").strip() for q in noisy_questions)

    last_line = clean.get("keep_last_statement", "Be sure to act as an information seeker only and not information provider")

    # Return as plain text (no markdown styling in the questions themselves)
    final_text = f"{header}{questions_txt}\n\n{base}\n\n{last_line}"
    return final_text

# =========================
# Main generation
# =========================
def main():
    load_dotenv()
    api_key = os.getenv("PERSONAL_OPENAI_KEY")
    if not api_key:
        raise ValueError("PERSONAL_OPENAI_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes = load_hash_history(HISTORY_FILE)

    generated = 0
    idx = START_INDEX

    while generated < NUM_PROMPTS:
        # Sample language & condition; seed RNG to reproduce per-user style if needed
        language = random.choice(LANGUAGES)
        condition_hint = random.choice(CONDITIONS)

        # Generate clean JSON from model
        data = call_openai(client, language, condition_hint)

        # Validate structure
        profile = data.get("profile", {})
        context = data.get("context", "")
        example_questions = data.get("example_questions", [])
        keep_last = data.get("keep_last_statement", "").strip()

        if not isinstance(example_questions, list) or not example_questions:
            # Try again
            continue
        if keep_last != "Be sure to act as an information seeker only and not information provider":
            # enforce exactness; retry
            continue
        if profile.get("language", "").lower() != language.lower():
            # Light fix-up: enforce target language in profile field
            profile["language"] = language
            data["profile"] = profile

        # Rough language sanity (optional)
        if not rough_language_check(example_questions, language):
            # You could re-ask the model here; for now, skip
            continue

        # Dedup on essence (clean items)
        essence = {
            "profile": profile,
            "context": context,
            "example_questions_clean": example_questions,
        }
        essence_hash = sha256(normalize_for_hash(essence))
        if essence_hash in seen_hashes:
            # try a different sample
            continue

        # Style profile & corruption
        rng = random.Random(time.time_ns())
        style_prof = sample_style_profile(rng)
        noisy = [apply_style_profile(q, style_prof) for q in example_questions]

        # Build final text prompt following structure/insertion rules
        final_text = build_final_prompt_text(
            {"profile": profile, "context": context, "keep_last_statement": keep_last},
            noisy,
            language,
        )

        # Save files
        txt_path = PROMPT_DIR / f"prompt{idx}.txt"
        json_path = JSON_DIR / f"prompt{idx}.json"

        with txt_path.open("w", encoding="utf-8") as f:
            f.write(final_text)

        bundle = {
            "profile": profile,
            "context": context,
            "example_questions_clean": example_questions,
            "example_questions_noisy": noisy,
            "style_profile": style_prof,
            "_raw_model_json": data.get("_raw_model_json", data),
            "_essence_hash": essence_hash,
            "_language": language,
            "_condition_hint": condition_hint,
            "_model": MODEL,
        }
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(bundle, jf, ensure_ascii=False, indent=2)

        # Update history
        append_hash_history(HISTORY_FILE, essence_hash)
        seen_hashes.add(essence_hash)

        print(f"Generated and saved: {txt_path} | {json_path}")
        idx += 1
        generated += 1

    print(f"Successfully generated {generated} unique prompts starting from prompt{START_INDEX}.txt")

if __name__ == "__main__":
    main()
