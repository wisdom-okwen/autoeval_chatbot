import os
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PERSONAL_OPENAI_KEY")
if not api_key:
    raise ValueError("PERSONAL_OPENAI_KEY environment variable is not set.")

client = OpenAI(api_key=api_key)

PROMPT_DIR = "/playpen-ssd/wokwen/projects/autoeval_chatbot/prompts/"
HISTORY_FILE = "/playpen-ssd/wokwen/projects/autoeval_chatbot/prompt_history.txt"

os.makedirs(PROMPT_DIR, exist_ok=True)

NUM_PROMPTS = 1
START_INDEX = 530

languages = ["English", "French", "English", "Spanish", "English", "Portuguese", "English", "Swahili", "Dutch", "English"]

conditions = [
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
    "feeling socially isolated"
]

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
"""

def load_prompt_history():
    if not os.path.exists(HISTORY_FILE):
        return set()
    with open(HISTORY_FILE, "r", encoding="utf-8") as file:
        return set(file.read().split("\n\n"))

def save_prompt_to_history(prompt):
    with open(HISTORY_FILE, "a", encoding="utf-8") as file:
        file.write(prompt + "\n\n")

def generate_prompt():
    language = random.choice(languages)
    condition = random.choice(conditions)
    history = load_prompt_history()

    prompt_request = (
        "Generate a unique conversation prompt for a user seeking information about PrEP. "
        "The user is in a unique situation because of a personal condition or challenge. "
        f"Example condition: 'Spiritual attacks and supernatural controls over your my life'. "
        "Describe their situation, how the condition may affect their mindset, emotions, or barriers to PrEP access. "
        "Generate their profile including nationality, age, gender, socio-economic status, primary concern about PrEP, and language of conversation. "
        "Keep prompts emotionally grounded, reflecting possible stigma, fear, or hope.\n\n"
        f"**The user can only speak {language}**. As such, example questions should be in this language.\n"
        "**Follow the structure in this example strictly else your response would be thrown out the window:**\n"
        f"{EXAMPLE_PROMPT_EDGE_CASE}\n\n"
        "**Always include the last statement from the example**"
        f"**Inlcude this statement in the after the forth line: My situation is unique because I have/I'm 'Spiritual attacks and supernatural controls over your my life'**"
        "**Make sure example questions are not styled, formatted or in markdown format. They should be plain text, single sentence.**"
        "**IMPORTANT: The user's example questions should include frequent typing errors, grammatical mistakes, poor punctuation, and inconsistent spacing. Some questions should be emotional, abrupt, or lack proper structure. DO NOT correct the user's language.**"
        "Ensure each prompt shows how the user's condition affects their thinking or questions about PrEP.\n"
        f"Exclude reused prompts from this history: {history}\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt_request}],
            max_tokens=450,
            temperature=0.85
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating prompt: {str(e)}"

if __name__ == '__main__':
    generated_count = 0
    for i in range(START_INDEX, START_INDEX + NUM_PROMPTS):
        prompt_text = generate_prompt()
        save_prompt_to_history(prompt_text)

        file_path = os.path.join(PROMPT_DIR, f"prompt{i}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(prompt_text)

        print(f"Generated and saved: {file_path}")
        generated_count += 1

    print(f"Successfully generated {generated_count} unique prompts starting from prompt{START_INDEX}.txt")