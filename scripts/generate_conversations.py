
import os
import csv
import re
import json
import random
import openai
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('PERSONAL_OPENAI_KEY')
if not api_key:
    raise ValueError('PERSONAL_OPENAI_KEY not found in .env')
client = openai.OpenAI(api_key=api_key)

BASE_DIR = Path(os.path.dirname(os.path.dirname(__file__)))
PROMPTS_DIR = BASE_DIR / 'prompts'
CONV_DIR = BASE_DIR / 'conversations'
os.makedirs(CONV_DIR, exist_ok=True)

# Noise probability (55% of user turns noisy). We'll apply after generating clean text.
NOISY_PROPORTION = 0.55  # proportion of user turns that will be intentionally corrupted

# Language regex helper
LANG_RE = re.compile(r'"language"\s*:\s*"(.*?)"', re.IGNORECASE)

# Helper to extract questions from prompt file
QUESTION_PATTERN = re.compile(r'^Turn \d+: (.+)$', re.MULTILINE)

def extract_questions(prompt_path: Path):
    content = prompt_path.read_text(encoding='utf-8')
    match = re.search(r"You may start by asking:\s*\n?\"(.+?)\"", content)
    if match:
        return [match.group(1).strip()]
    return []

def extract_profile_and_context(prompt_path: Path):
    content = prompt_path.read_text(encoding='utf-8')
    # Attempt to also locate JSON companion to get structured language
    json_candidate = prompt_path.with_suffix('.json')
    language = None
    if json_candidate.exists():
        try:
            data = json.loads(json_candidate.read_text(encoding='utf-8'))
            language = data.get('profile', {}).get('language')
        except Exception:
            pass
    if not language:
        # fallback heuristic
        m = re.search(r'"language"\s*:\s*"(.*?)"', content, re.IGNORECASE)
        language = m.group(1) if m else 'English'
    # crude persona/context extraction
    persona_block = []
    for line in content.splitlines():
        if line.strip().startswith('You may start by asking:'):
            break
        persona_block.append(line.strip())
    persona = ' '.join(persona_block[-30:])  # last lines more specific
    return persona, language

def get_gpt_response(user_input):
    system_prompt = (
        "You are a friendly, inclusive, and knowledgeable HIV prevention and PrEP counseling assistant. "
        "Respond in clear, simple, and supportive language. "
        "Always answer as if you are talking to a real person, and never provide medical advice—just information. "
        "Be concise, avoid jargon, and use a conversational tone. "
        "If the user expresses fear, stigma, or mental health concerns, respond with empathy and suggest professional support resources. "
        "If the user asks about PrEP, always mention both oral and injectable options unless they specify one. "
        "Do not make recommendations or value judgments. "
        "If the user asks about side effects, effectiveness, or cost, provide neutral, factual information. "
        "If the user asks about stigma, privacy, or relationships, be supportive and nonjudgmental. "
        "If the user asks about travel, access, or insurance, provide practical information. "
        "Always answer only the question asked, and keep each response under 120 words."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=350,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {e}]"

def maybe_corrupt(text: str, style_seed: int) -> str:
    # lightweight noise: random casing + minor punctuation & typo injection
    rng = random.Random(style_seed)
    if not text:
        return text
    # Randomly lower-case start or whole
    if rng.random() < 0.25:
        text = text.lower()
    elif rng.random() < 0.5 and text[0].isalpha():
        text = text[0].lower() + text[1:]
    # random char swaps/typos
    chars = list(text)
    for i,ch in enumerate(chars):
        if ch.isalpha() and rng.random() < 0.04:
            chars[i] = ch*2
        if ch.isalpha() and rng.random() < 0.03:
            chars[i] = ch.lower() if ch.isupper() else ch.upper()
    text = ''.join(chars)
    if rng.random() < 0.2:
        text = re.sub(r'[.,!?]','',text)
    if rng.random() < 0.18:
        text += ' ' + random.choice(['uh','um','idk','pls','??'])
    return text

def simulate_conversation(prompt_file: Path, conv_file: Path):
    questions = extract_questions(prompt_file)
    persona, language = extract_profile_and_context(prompt_file)
    content = prompt_file.read_text(encoding='utf-8')
    followup_match = re.search(r'A possible follow-up you might ask later:\n\"(.+?)\"', content)
    followup_guidance = followup_match.group(1).strip() if followup_match else "ask something more specific about access, cost, side effects, adherence, or stigma."
    if not questions:
        print(f"No starter question found in {prompt_file.name}, skipping.")
        return
    starter = questions[0]
    asked_questions = set([starter.lower()])
    noisy_turn_indices = set(random.sample(range(1,31), k=int(round(30*NOISY_PROPORTION))))
    with conv_file.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['turn','user','bot','language','has_error'])
        user_q = starter
        for turn in range(1,31):
            # Decide if this user turn is noisy
            display_user_q = user_q
            has_error = turn in noisy_turn_indices
            if has_error:
                display_user_q = maybe_corrupt(user_q, style_seed=hash(user_q) & 0xffffffff)
            bot_response = get_gpt_response(f"Persona: {persona}\nUser question: {display_user_q}")
            writer.writerow([turn, display_user_q, bot_response, language, int(has_error)])
            if turn == 30:
                break
            # Generate next user question
            gen_prompt = (
                "Generate the NEXT user question only (no prefix) continuing a 30-turn conversation about HIV prevention/PrEP. "
                "User never thanks explicitly, keeps direct, informal, emotionally authentic. Avoid repeating earlier phrasing. "
                f"Persona context: {persona}\n"
                f"Previous user question: {user_q}\n"
                f"Assistant reply: {bot_response}\n"
                f"Starter question was: {starter}\n"
                f"Follow-up hint: {followup_guidance}\n"
                "Constraints: under 25 words, no greetings, no closing, no lists, one sentence. Output only the question."
            )
            try:
                resp = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[{"role":"user","content":gen_prompt}],
                    max_tokens=60,
                    temperature=0.9
                )
                candidate = resp.choices[0].message.content.strip()
            except Exception as e:
                candidate = f"error next q: {e}"
            cleaned = re.sub(r'\s+',' ', candidate)
            # Avoid repetition
            if cleaned.lower() in asked_questions:
                cleaned += ' ?'
            asked_questions.add(cleaned.lower())
            user_q = cleaned

def main():
    prompt_files = [f"prompt_{i:03d}.txt" for i in range(0,10)]
    for pf in prompt_files:
        prompt_path = PROMPTS_DIR / pf
        if not prompt_path.exists():
            print(f"Prompt file {pf} missing, skipping")
            continue
        conv_path = CONV_DIR / pf.replace('prompt_','conv_').replace('.txt','.csv')
        print(f"Simulating conversation for {pf}")
        simulate_conversation(prompt_path, conv_path)
    print("Conversations generated for first 10 prompts.")

if __name__ == "__main__":
    main()
