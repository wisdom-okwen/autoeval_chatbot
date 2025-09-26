import os
import re
import json
import time
import random
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Config
# =========================
PROMPT_DIR = Path("/playpen-ssd/wokwen/projects/autoeval_chatbot/new_prompts/")
JSON_DIR   = PROMPT_DIR
HISTORY_FILE = Path("/playpen-ssd/wokwen/projects/autoeval_chatbot/prompt_history_hashes.txt")   # whole-prompt essence
STARTER_HISTORY_FILE = Path("/playpen-ssd/wokwen/projects/autoeval_chatbot/starter_history_hashes.txt")  # only starters

# Phase 1: generate only 10 prompts (000-009). Later scaling will adjust NUM_PROMPTS and START_INDEX.
NUM_PROMPTS = 10
START_INDEX = 0
MODEL = "gpt-4o-mini"
MAX_TOKENS = 400
TEMPERATURE = 0.9
TOP_P = 0.95
FREQ_PENALTY = 0.2

# Language weighting: target English at ~30%. We'll sample using explicit weights instead of list repetition.
LANGUAGE_WEIGHTS = {
    "English": 0.30,
    "French": 0.14,
    "Spanish": 0.14,
    "Portuguese": 0.10,
    "Swahili": 0.10,
    "Dutch": 0.10,
    "German": 0.06,
    "Arabic": 0.06
}
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

# Example structure block (we will programmatically inject starter/follow-up)
# Minimal skeleton for final formatting will be built dynamically; removing hard-coded English example.
EXAMPLE_PROMPT_SKELETON = (
    "You are an information seeker with a complex life context impacting access to PrEP. "
    "Show emotional authenticity (anxiety, hesitation, stigma, hope) without providing information yourself. "
    "You only ask questions.\n\n"
    "You may start by asking:\n\"{STARTER_QUESTION}\"\n\n"
    "A possible follow-up you might ask later:\n\"{FOLLOWUP_QUESTION}\"\n\n"
    "Keep questions specific, realistic, context-grounded.\n\n"
    "Be sure to act as an information seeker only and not information provider"
)

# =========================
# Utilities: history & hashing
# =========================
def _load_hashes(path: Path) -> set:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

def _append_hash(path: Path, h: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(h + "\n")

def normalize_for_hash(d: Dict[str, Any]) -> str:
    essence = {
        "profile": d.get("profile", {}),
        "context": d.get("context", ""),
        "example_questions_clean": d.get("example_questions_clean", []),
        "starter_clean": d.get("starter_clean", ""),
    }
    return json.dumps(essence, sort_keys=True, ensure_ascii=False).lower()

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# =========================
# Noise injection
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
# Minimal language check (very rough)
# =========================
def rough_language_check(texts: List[str], target_language: str) -> bool:
    if target_language.lower() == "english":
        sample = " ".join(texts).lower()
        hits = sum(w in sample for w in ["the", "and", "i", "to", "you", "is"])
        return hits >= 2
    return True

# =========================
# OpenAI request & formatting
# =========================
def weighted_language_choice() -> str:
    r = random.random()
    cumulative = 0.0
    for lang, w in LANGUAGE_WEIGHTS.items():
        cumulative += w
        if r <= cumulative:
            return lang
    # fallback last
    return list(LANGUAGE_WEIGHTS.keys())[-1]

def build_user_instruction(language: str, condition_hint: str) -> str:
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
        "  example_questions: array of 4-6 plain sentences in TARGET_LANGUAGE (no markdown, no emojis, clean grammar),\n"
        "  keep_last_statement: string  // must exactly be: 'Be sure to act as an information seeker only and not information provider'\n"
        "Rules:\n"
        "- The example_questions MUST be clean and grammatical (we will add typos later).\n"
        "- Make the first 2 questions different in theme and wording.\n"
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
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        frequency_penalty=FREQ_PENALTY,
    )
    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
    except Exception as e:
        content2 = content.strip().strip("`")
        try:
            data = json.loads(content2)
        except Exception:
            raise ValueError(f"Model did not return valid JSON. Raw:\n{content}") from e
    data["_raw_model_json"] = content
    return data

def build_prompt_text_dynamic(clean: Dict[str, Any], language: str,
                              starter_noisy: str, followup_noisy: str,
                              noisy_list: List[str]) -> str:
    # Build header
    header = (
        f"**The user can only speak {language}. As such, example questions should be in this language.**\n"
        f"Profile: {json.dumps(clean.get('profile', {}), ensure_ascii=False)}\n\n"
        f"Context:\n{clean.get('context', '').strip()}\n\n"
        "Example questions (plain text, noisy):\n"
    )
    questions_txt = "\n".join(q.replace("\n", " ").strip() for q in noisy_list)

    # Build example block with dynamic starter and follow-up
    base = EXAMPLE_PROMPT_SKELETON.replace("{STARTER_QUESTION}", starter_noisy)
    base = base.replace("{FOLLOWUP_QUESTION}", followup_noisy)

    last_line = clean.get("keep_last_statement", "Be sure to act as an information seeker only and not information provider")
    return f"{header}{questions_txt}\n\n{base}\n\n{last_line}"

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
    STARTER_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    seen_essences = _load_hashes(HISTORY_FILE)
    seen_starters = _load_hashes(STARTER_HISTORY_FILE)

    generated = 0
    idx = START_INDEX

    while generated < NUM_PROMPTS:
        language = weighted_language_choice()
        condition = random.choice(CONDITIONS)

        data = call_openai(client, language, condition)

        profile = data.get("profile", {})
        context = data.get("context", "")
        clean_qs: List[str] = data.get("example_questions", []) or []
        keep_last = (data.get("keep_last_statement") or "").strip()

        if not clean_qs or keep_last != "Be sure to act as an information seeker only and not information provider":
            continue
        if profile.get("language", "").lower() != language.lower():
            profile["language"] = language

        if not rough_language_check(clean_qs, language):
            continue

        # Choose a unique starter (clean); retry within the set if starter already used
        rng = random.Random(time.time_ns())
        candidate_indices = list(range(len(clean_qs)))
        rng.shuffle(candidate_indices)
        starter_idx: Optional[int] = None
        for i in candidate_indices:
            starter_clean = clean_qs[i].strip()
            starter_hash = sha256(starter_clean.lower())
            if starter_hash not in seen_starters:
                starter_idx = i
                break
        if starter_idx is None:
            # all starters seen before → skip and regenerate a new prompt
            continue

        # Pick a follow-up from remaining questions if any, else reuse another unique index
        remaining = [j for j in range(len(clean_qs)) if j != starter_idx]
        follow_idx = remaining[0] if remaining else starter_idx

        # Style + noise
        style_prof = sample_style_profile(rng)
        noisy_qs = [apply_style_profile(q, style_prof) for q in clean_qs]
        starter_noisy = noisy_qs[starter_idx]
        followup_noisy = noisy_qs[follow_idx]

        # Dedup: include starter_clean in essence to increase variability constraint
        essence_payload = {
            "profile": profile,
            "context": context,
            "example_questions_clean": clean_qs,
            "starter_clean": clean_qs[starter_idx],
        }
        essence_hash = sha256(normalize_for_hash(essence_payload))
        if essence_hash in seen_essences:
            # try again for a different sample
            continue

        # Compose final prompt text with dynamic starter/follow-up
        final_text = build_prompt_text_dynamic(
            {"profile": profile, "context": context, "keep_last_statement": keep_last},
            language,
            starter_noisy=starter_noisy,
            followup_noisy=followup_noisy,
            noisy_list=noisy_qs,
        )

        # Save files
        txt_path = PROMPT_DIR / f"prompt_{idx:03d}.txt"
        json_path = JSON_DIR / f"prompt_{idx:03d}.json"
        txt_path.write_text(final_text, encoding="utf-8")

        bundle = {
            "profile": profile,
            "context": context,
            "example_questions_clean": clean_qs,
            "example_questions_noisy": noisy_qs,
            "starter_index": starter_idx,
            "starter_clean": clean_qs[starter_idx],
            "starter_noisy": starter_noisy,
            "followup_index": follow_idx,
            "followup_clean": clean_qs[follow_idx],
            "followup_noisy": followup_noisy,
            "style_profile": style_prof,
            "_raw_model_json": data.get("_raw_model_json", data),
            "_essence_hash": essence_hash,
            "_starter_hash": sha256(clean_qs[starter_idx].lower()),
            "_language": language,
            "_condition_hint": condition,
            "_model": MODEL,
        }
        json_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

        # Update histories
        _append_hash(HISTORY_FILE, essence_hash)
        seen_essences.add(essence_hash)

        starter_hash = bundle["_starter_hash"]
        _append_hash(STARTER_HISTORY_FILE, starter_hash)
        seen_starters.add(starter_hash)

        print(f"Generated and saved: {txt_path.name} | starter='{clean_qs[starter_idx]}' | lang={language}")
        idx += 1
        generated += 1
    print(f"Successfully generated {generated} unique prompts starting from prompt_{START_INDEX:03d}.txt")

if __name__ == "__main__":
    main()
