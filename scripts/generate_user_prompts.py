import os
import random
import openai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('OPENAI_API_KEY not found in .env')
openai.api_key = api_key

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

def generate_prompts(n=100, output_file='user_prompts.csv'):
    profiles = generate_user_profiles(n)
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
    os.makedirs(prompts_dir, exist_ok=True)
    for idx, profile in enumerate(profiles, 1):
        if random.random() < 0.7:
            extra = " Use language as real humans do: you may use incomplete sentences, slang, misspellings, abbreviations, emojis, or broken words, just like people do when texting. Don't worry about perfect grammar."
        else:
            extra = ""
        prompt = prompt_template.format(
            age=profile[0],
            race=profile[1],
            country=profile[2],
            ethnicity=profile[3],
            gender=profile[4],
            sexual_orientation=profile[5],
            education_level=profile[6],
            socio_economic=profile[7],
            religion=profile[8],
            urban_rural=profile[9],
            marital_status=profile[10],
            language=profile[11]
        ) + extra
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=200
            )
            user_message = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            user_message = f"[Error generating prompt: {e}]"
        # Save each prompt to a separate file
        filename = f"prompt_{idx:03d}.txt"
        with open(os.path.join(prompts_dir, filename), 'w', encoding='utf-8') as pf:
            pf.write(f"# Profile: {profile}\n\n{user_message}\n")
    print(f"Generated {n} user prompts and saved to {prompts_dir}")

if __name__ == "__main__":
    generate_prompts()
