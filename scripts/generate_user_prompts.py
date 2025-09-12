import os
import random
import openai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('PERSONAL_OPENAI_KEY')
if not api_key:
    raise ValueError('PERSONAL_OPENAI_KEY not found in .env')
client = openai.OpenAI(api_key=api_key)


# Languages and their weights (English more common)
languages = [
    ('English', 0.5),
    ('Spanish', 0.1),
    ('French', 0.08),
    ('Swahili', 0.07),
    ('Hindi', 0.07),
    ('Mandarin', 0.07),
    ('Arabic', 0.05),
    ('Portuguese', 0.03),
    ('Russian', 0.03)
]
language_choices = [lang for lang, weight in languages for _ in range(int(weight*100))]

# User backgrounds
ages = ['teenager', 'young adult', 'adult', 'middle-aged', 'senior']
races = ['Black', 'White', 'Asian', 'Latino', 'Indigenous', 'Mixed', 'Arab']
socio_economic = ['low income', 'middle income', 'high income']
countries = ['USA', 'Nigeria', 'India', 'Brazil', 'France', 'China', 'South Africa', 'Kenya', 'Mexico', 'Egypt']
ethnicities = ['African American', 'Hispanic', 'Han Chinese', 'Punjabi', 'Zulu', 'French', 'Yoruba', 'Arab', 'Russian', 'Indigenous Australian']
genders = ['female', 'male', 'non-binary', 'transgender', 'genderqueer']
sexual_orientations = ['heterosexual', 'homosexual', 'bisexual', 'asexual', 'pansexual', 'queer']
education_levels = ['no formal education', 'primary', 'secondary', 'college', 'graduate', 'postgraduate']

# Additional backgrounds
religions = ['Christian', 'Muslim', 'Hindu', 'Buddhist', 'Jewish', 'Atheist', 'Agnostic', 'Traditional']
urban_rural = ['urban', 'rural', 'suburban']
marital_status = ['single', 'married', 'divorced', 'widowed', 'in a relationship']

# Prompt template
prompt_template = """
You are a {age} {race} person from {country} of {ethnicity} ethnicity, {gender}, {sexual_orientation}, {education_level} education, {socio_economic} background, {religion} faith, living in a(n) {urban_rural} area, and {marital_status}. You want to use the ShesPrEPared chatbot to learn about HIV prevention. Write a message in {language} that you might send to the chatbot, reflecting your background and concerns. Be natural and informal, as if texting a real person. Sometimes, use incomplete sentences, slang, misspellings, or broken words, just like real people do when chatting online. Avoid repeating previous prompts.
"""

def generate_user_profiles(n):
    profiles = set()
    while len(profiles) < n:
        profile = (
            random.choice(ages),
            random.choice(races),
            random.choice(countries),
            random.choice(ethnicities),
            random.choice(genders),
            random.choice(sexual_orientations),
            random.choice(education_levels),
            random.choice(socio_economic),
            random.choice(religions),
            random.choice(urban_rural),
            random.choice(marital_status),
            random.choice(language_choices)
        )
        profiles.add(profile)
    return list(profiles)

def random_communication_style():
    styles = [
        'formal', 'casual', 'slang', 'emoji-heavy', 'broken English', 'abbreviated', 'typos', 'code-switching', 'direct', 'hesitant', 'emotional', 'neutral'
    ]
    return random.choice(styles)

def random_emotional_state():
    states = [
        'curious', 'anxious', 'skeptical', 'hopeful', 'confused', 'determined', 'shy', 'outspoken', 'cautious', 'excited', 'nervous', 'doubtful', 'empowered', 'overwhelmed'
    ]
    return random.choice(states)

def random_primary_concern():
    concerns = [
        'effectiveness', 'side effects', 'cost', 'stigma', 'access', 'adherence', 'long-term safety', 'relationships', 'privacy', 'travel', 'community perception', 'misinformation', 'support', 'insurance', 'doctor trust', 'cultural beliefs'
    ]
    return random.choice(concerns)

def generate_question_sequence(profile, style, emotion, concern, language):
    # Generate a sequence of 3-5 questions, escalating or shifting focus
    base_questions = {
        'effectiveness': [
            "How well does PrEP actually work?",
            "Is it 100% effective or are there cases where it fails?",
            "Can you give real examples or stats?"
        ],
        'side effects': [
            "What side effects should I expect?",
            "Are there any long-term risks?",
            "What if I get a bad reaction?"
        ],
        'cost': [
            "Is PrEP expensive?",
            "What if I can't afford it?",
            "Does insurance cover it?"
        ],
        'stigma': [
            "Will people judge me for using PrEP?",
            "How do I deal with stigma?",
            "What if my family finds out?"
        ],
        'access': [
            "How do I get PrEP?",
            "Do I need a prescription?",
            "Is it available in my area?"
        ],
        'adherence': [
            "What if I forget a dose?",
            "Is it okay to miss sometimes?",
            "How do people remember to take it every day?"
        ],
        'long-term safety': [
            "Is it safe to take PrEP for years?",
            "What happens if I stop?",
            "Are there alternatives?"
        ],
        'relationships': [
            "Should I tell my partner I'm on PrEP?",
            "How do I talk about it with them?",
            "Will they think I don't trust them?"
        ],
        'privacy': [
            "Can I get PrEP without anyone knowing?",
            "Will it show up on my records?",
            "How do I keep it private?"
        ],
        'travel': [
            "Can I travel with PrEP?",
            "What if I run out abroad?",
            "Are there countries where it's not allowed?"
        ],
        'community perception': [
            "What do people in my community think about PrEP?",
            "Is there a lot of misinformation?",
            "How do I find support?"
        ],
        'misinformation': [
            "I've heard a lot of rumors about PrEP. What's true?",
            "How do I know what to believe?",
            "Where can I get real info?"
        ],
        'support': [
            "Are there support groups for people on PrEP?",
            "How do I connect with others?",
            "Is there counseling available?"
        ],
        'insurance': [
            "Does insurance cover PrEP?",
            "What if I lose my insurance?",
            "Are there programs to help?"
        ],
        'doctor trust': [
            "How do I know I can trust my doctor about PrEP?",
            "What if they judge me?",
            "Can I get a second opinion?"
        ],
        'cultural beliefs': [
            "Is PrEP accepted in my culture?",
            "What if my beliefs conflict with taking it?",
            "How do others handle this?"
        ]
    }
    # Pick 3-5 questions, shuffle, and apply style/emotion
    questions = base_questions.get(concern, ["What should I know about PrEP?"])
    random.shuffle(questions)
    n_q = random.randint(3, 5)
    selected = questions[:n_q]
    # Style and emotion tweaks
    for i in range(len(selected)):
        if style == 'slang':
            selected[i] = selected[i].replace('you', 'u').replace('your', 'ur').replace('are', 'r').replace('with', 'w/').replace('about', 'abt')
        if style == 'emoji-heavy':
            selected[i] += ' 🤔' if i % 2 == 0 else ' 🙏'
        if style == 'broken English':
            def break_word(w):
                if len(w) <= 2:
                    return w
                return w[:random.randint(2, len(w))]
            selected[i] = ' '.join([break_word(w) for w in selected[i].split()])
        if style == 'abbreviated':
            selected[i] = selected[i][:random.randint(10, max(15, len(selected[i])))] + '...'
        if style == 'typos':
            if len(selected[i]) > 8:
                pos = random.randint(1, len(selected[i])-2)
                selected[i] = selected[i][:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + selected[i][pos+1:]
        if style == 'code-switching' and language != 'English':
            selected[i] = f"[{language}] " + selected[i]
        if emotion == 'anxious':
            selected[i] = selected[i] + ' (I’m kinda worried tbh)'
        if emotion == 'skeptical':
            selected[i] = selected[i] + ' (Are you sure tho?)'
        if emotion == 'hopeful':
            selected[i] = selected[i] + ' (I hope this works!)'
        if emotion == 'shy':
            selected[i] = 'Um... ' + selected[i]
        if emotion == 'outspoken':
            selected[i] = selected[i].upper()
    return selected

def generate_prompts(n=10, output_file='user_prompts.csv'):
    profiles = generate_user_profiles(n)
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
    os.makedirs(prompts_dir, exist_ok=True)
    for idx, profile in enumerate(profiles, 1):
        style = random_communication_style()
        emotion = random_emotional_state()
        concern = random_primary_concern()
        language = profile[11]
        persona = f"Nationality: {profile[2]}\nAge: {profile[0]}\nGender: {profile[4].capitalize()}\nSocio-economic background: {profile[7].capitalize()}\nEducation Level: {profile[6].capitalize()}\nEthnicity: {profile[3]}\nReligion: {profile[8]}\nUrban/Rural: {profile[9]}\nMarital Status: {profile[10]}\nSexual Orientation: {profile[5]}\nLanguage of conversation: {language}"
        gpt_prompt = (
            f"Given the following user profile, generate a vivid, realistic scenario for a chatbot conversation about HIV prevention and PrEP. "
            f"The output should include:\n"
            f"1. A scenario description (1-2 sentences) that sets the user's motivation and emotional state.\n"
            f"2. A sentence like: 'You should start by asking: ...' with a realistic, profile-specific opening question.\n"
            f"3. 1-2 sentences of guidance for how the user might proceed in follow-up turns (e.g., 'As the conversation progresses, you might ask about...').\n"
            f"4. End with: 'Be sure to act as an information seeker only and not information provider. Questions should be specific to your profile.'\n"
            f"The language and concerns should match the user's background.\n"
            f"\nUser profile:\n{persona}\n\nPrimary concern: {concern}\nCommunication style: {style}\nEmotional state: {emotion}\n"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": gpt_prompt}],
                max_tokens=400,
                temperature=0.8
            )
            prompt_text = response.choices[0].message.content.strip()
        except Exception as e:
            prompt_text = f"[Error generating prompt: {e}]"
        filename = f"prompt_{idx:03d}.txt"
        with open(os.path.join(prompts_dir, filename), 'w', encoding='utf-8') as pf:
            pf.write(f"{prompt_text}\n")
    print(f"Generated {n} user prompts and saved to {prompts_dir}")

if __name__ == "__main__":
    generate_prompts()
