
import os
import csv
import re
import openai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('PERSONAL_OPENAI_KEY')
if not api_key:
    raise ValueError('PERSONAL_OPENAI_KEY not found in .env')
client = openai.OpenAI(api_key=api_key)

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
CONV_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'conversations')
os.makedirs(CONV_DIR, exist_ok=True)

# Helper to extract questions from prompt file
QUESTION_PATTERN = re.compile(r'^Turn \d+: (.+)$', re.MULTILINE)

def extract_questions(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Find the line with 'You should start by asking:'
    import re
    match = re.search(r"You should start by asking:\s*[\"']?(.+?)[\"']?(?:\n|$)", content)
    if match:
        return [match.group(1).strip()]
    return []

def extract_persona(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    persona = []
    for line in lines:
        if line.strip().startswith('--- CONVERSATION TURNS ---'):
            break
        persona.append(line.strip())
    return ' '.join(persona)

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

def simulate_conversation(prompt_file, conv_file):
    questions = extract_questions(prompt_file)
    persona = extract_persona(prompt_file)
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_content = f.read()
    # Extract follow-up guidance
    import re
    followup_match = re.search(r'As the conversation progresses, you might (.+?)(?:\.|\n)', prompt_content)
    followup_guidance = followup_match.group(1).strip() if followup_match else "ask more about your concerns or related topics."
    with open(conv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['turn', 'user', 'bot'])
        # Start with the initial question
        if not questions:
            return
        user_q = questions[0]
        for i in range(1, 31):
            user_input = f"[{persona}] {user_q}"
            bot_response = get_gpt_response(user_input)
            writer.writerow([i, user_q, bot_response])
            # For next turn, generate a new user question using GPT
            if i < 30:
                followup_prompt = (
                    f"Given the following user profile and the last bot response, generate the next user question for a chatbot conversation about HIV prevention and PrEP. "
                    f"The user should continue as an information seeker, following this guidance: {followup_guidance}. "
                    f"Be direct and to the point. Do not use polite filler phrases like 'Thank you for that information!' or 'I appreciate your help.' Avoid greetings, closings, and small talk. Use broken English, poor grammar, or spelling mistakes where possible. Be realistic, natural, and specific to the user's background.\n"
                    f"User profile: {persona}\n"
                    f"Last user question: {user_q}\n"
                    f"Bot response: {bot_response}\n"
                    f"Next user question:"
                )
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": followup_prompt}],
                        max_tokens=80,
                        temperature=0.8
                    )
                    user_q = response.choices[0].message.content.strip()
                except Exception as e:
                    user_q = f"[Error generating next user question: {e}]"

def main():
    prompt_files = sorted([f for f in os.listdir(PROMPTS_DIR) if f.endswith('.txt')])
    for pf in prompt_files:
        prompt_path = os.path.join(PROMPTS_DIR, pf)
        conv_name = pf.replace('prompt_', 'conv_').replace('.txt', '.csv')
        conv_path = os.path.join(CONV_DIR, conv_name)
        print(f"Simulating conversation for {pf} -> {conv_name}")
        simulate_conversation(prompt_path, conv_path)
    print("All conversations generated.")

if __name__ == "__main__":
    main()
