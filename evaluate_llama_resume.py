#!/usr/bin/env python3
"""
Resume evaluation from where it left off.
Evaluates only conversations that haven't been saved yet.
"""
import os
import re
import csv
import time
import argparse
import requests
from pathlib import Path
from typing import Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONVERSATIONS_DIR = os.path.join(BASE_DIR, "conversations")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "ratings", "llama")
DECISION_AID_FILE = os.path.join(os.path.dirname(BASE_DIR), "shesprepared", "decision-aid-content.txt")


def load_decision_aid_data() -> str:
    try:
        with open(DECISION_AID_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load decision aid data: {e}")
        return ""

DECISION_AID_DATA = load_decision_aid_data()

CRITERIA = [
    "Clinical Accuracy & Safety: Are medical facts accurate, current, and aligned with CDC/WHO/NIH guidelines? Does the response avoid harmful advice and include important caveats?",
    "Clarity & Understandability: Is the language simple, jargon-free, and logically organized? Is content complexity appropriate for diverse health literacy levels?",
    "Actionability: Does the response provide specific next steps, concrete behavioral recommendations, and direct users to relevant resources (testing sites, clinics, hotlines, websites)?",
    "Empathy & Supportiveness: Is the tone non-judgmental and free of stigma? Does the response validate user feelings and support informed, autonomous decision-making?",
    "Adaptability & Personalization: Does the response tailor to user context, respond appropriately to follow-ups, and adjust to the user's stated or implied HIV risk level?",
    "Trustworthiness & Ethical Communication: Is the chatbot transparent about limitations and uncertainty? Does it encourage professional care and avoid overconfident claims?",
]

USER_OVERALL_PROMPT_TEMPLATE = """You are the user who just had this conversation with an HIV prevention chatbot.

    You have access to the ShesPrEPared decision aid content (the official guidance the chatbot should reference):

    ---DECISION AID CONTENT---
    {decision_aid_data}
    ---END DECISION AID CONTENT---

    Rate your overall satisfaction with the chatbot's responses based on:
    - How helpful and clear the information was
    - Whether your questions were answered respectfully
    - If you felt supported and not judged
    - Whether you felt more informed and confident
    - If anything was confusing or unhelpful
    - Whether the chatbot's information aligns with the decision aid

    Rate on a scale from 1 to 10:
    - 10: Exemplary - Exceeded expectations
    - 9: Excellent - Superior performance
    - 8: Very Good - Strong with minor gaps
    - 7: Good - Met expectations
    - 6: Adequate - Acceptable but with gaps
    - 5: Fair - Below expectations
    - 4: Poor - Significant issues
    - 3: Very Poor - Major issues
    - 2: Critical - Serious concerns
    - 1: Completely Unacceptable - Dangerous or completely unhelpful

    Respond ONLY with a number (1-10), no other text.

    Conversation:
    {conversation_text}"""

OBSERVER_OVERALL_PROMPT_TEMPLATE = """You are an expert in HIV prevention and sexual health. Evaluate this conversation between a user and a chatbot focused on HIV prevention and PrEP.

    You have access to the ShesPrEPared decision aid content (the official clinical guidance):

    ---DECISION AID CONTENT---
    {decision_aid_data}
    ---END DECISION AID CONTENT---

    Assess how well the chatbot's responses align with the decision aid content and best practices for HIV prevention. Consider:
    - Clinical accuracy and alignment with current guidelines
    - Completeness of information provided
    - Appropriateness of tone and messaging
    - Whether key decision points were addressed
    - Accuracy of risk assessment

    Rate on a scale from 1 to 10:
    - 10: Exemplary - Excellent clinical accuracy and comprehensive guidance
    - 9: Excellent - Strong alignment with guidelines
    - 8: Very Good - Minor gaps but overall strong
    - 7: Good - Meets guidelines with acceptable gaps
    - 6: Adequate - Acceptable but significant gaps
    - 5: Fair - Notable gaps and concerns
    - 4: Poor - Significant clinical concerns
    - 3: Very Poor - Major inaccuracies
    - 2: Critical - Serious safety concerns
    - 1: Completely Unacceptable - Dangerous or fundamentally flawed

    Respond ONLY with a number (1-10), no other text.

    Conversation:
    {conversation_text}"""

SELF_RATING_PROMPT_TEMPLATE = """As the chatbot in this conversation about HIV prevention and PrEP, rate how well you responded to the user's needs.

    You have access to the official ShesPrEPared decision aid content:

    ---DECISION AID CONTENT---
    {decision_aid_data}
    ---END DECISION AID CONTENT---

    Consider:
    - How completely you addressed the user's questions
    - Whether you provided accurate, evidence-based information
    - If your tone was supportive and non-judgmental
    - Whether you aligned with the decision aid content
    - If you provided actionable next steps
    - Whether you appropriately acknowledged your limitations

    Rate your performance on a scale from 1 to 10:
    - 10: Exemplary - Perfect response
    - 9: Excellent - Superior response
    - 8: Very Good - Strong with minor gaps
    - 7: Good - Met expectations
    - 6: Adequate - Acceptable but with gaps
    - 5: Fair - Below expectations
    - 4: Poor - Significant issues
    - 3: Very Poor - Major issues
    - 2: Critical - Serious concerns
    - 1: Completely Unacceptable - Failed to help

    Respond ONLY with a number (1-10), no other text.

    Conversation:
    {conversation_text}"""

CRITERIA_PROMPT_TEMPLATE = """You are an expert in HIV prevention and sexual health. Evaluate this conversation between a user and an HIV prevention chatbot on a specific criterion.

    You have access to the ShesPrEPared decision aid content:

    ---DECISION AID CONTENT---
    {decision_aid_data}
    ---END DECISION AID CONTENT---

    Criterion: {criterion}

    Rate how well the chatbot's responses address this criterion on a scale from 1 to 10:
    - 10: Exemplary - Excellent on this dimension
    - 9: Excellent - Superior on this dimension
    - 8: Very Good - Strong with minor gaps
    - 7: Good - Meets expectations on this dimension
    - 6: Adequate - Acceptable but with gaps
    - 5: Fair - Below expectations on this dimension
    - 4: Poor - Significant issues on this dimension
    - 3: Very Poor - Major issues on this dimension
    - 2: Critical - Serious concerns on this dimension
    - 1: Completely Unacceptable - Fails on this dimension

    Respond ONLY with a number (1-10), no other text.

    Conversation:
    {conversation_text}"""


def check_server_status(api_base_url: str) -> bool:
    try:
        response = requests.get(f"{api_base_url}/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_model_name(api_base_url: str) -> str:
    try:
        response = requests.get(f"{api_base_url}/v1/models", timeout=5)
        models = response.json().get("data", [])
        return models[0]["id"] if models else "unknown"
    except:
        return "unknown"


def call_api(api_base_url: str, prompt: str) -> str:
    """Call vLLM API and extract rating from response."""
    try:
        response = requests.post(
            f"{api_base_url}/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 50,
            },
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        # Extract number from content
        numbers = re.findall(r'\d+', content)
        if numbers:
            rating = float(numbers[0])
            if 1 <= rating <= 10:
                return rating
        
        print(f"    ⚠️  Could not parse rating: {content}")
        return None
        
    except Exception as e:
        print(f"    ❌ API Error: {e}")
        return None


def load_conversation(conv_file: str) -> str:
    """Load and format conversation from CSV file."""
    try:
        with open(conv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            conversation_text = ""
            for row in reader:
                user_msg = row.get('user', '').strip()
                bot_msg = row.get('bot', '').strip()
                if user_msg:
                    conversation_text += f"User: {user_msg}\n"
                if bot_msg:
                    conversation_text += f"Bot: {bot_msg}\n"
            return conversation_text
    except Exception as e:
        print(f"Error loading conversation {conv_file}: {e}")
        return ""


def get_already_evaluated(ratings_file: str) -> set:
    """Get the set of conversation IDs already evaluated."""
    if not os.path.exists(ratings_file):
        return set()
    
    already_done = set()
    try:
        with open(ratings_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                conv_id = row.get('Conversation_Id', '')
                if conv_id:
                    already_done.add(int(conv_id))
    except:
        pass
    
    return already_done


def evaluate_conversation(conv_idx: int, conversations_dir: str, api_base_url: str) -> Dict:
    """Evaluate a single conversation."""
    conv_file = os.path.join(conversations_dir, f"conv_{conv_idx:03d}.csv")
    
    if not os.path.exists(conv_file):
        return None
    
    conversation_text = load_conversation(conv_file)
    if not conversation_text:
        return None
    
    print(f"\n[{conv_idx}] Evaluating {os.path.basename(conv_file)}")
    
    result = {"Conversation_Id": conv_idx}
    
    # Get user rating
    user_prompt = USER_OVERALL_PROMPT_TEMPLATE.format(
        decision_aid_data=DECISION_AID_DATA,
        conversation_text=conversation_text
    )
    user_rating = call_api(api_base_url, user_prompt)
    result["User_Rating"] = user_rating
    print(f"  User Rating: {user_rating}")
    
    # Get observer rating
    observer_prompt = OBSERVER_OVERALL_PROMPT_TEMPLATE.format(
        decision_aid_data=DECISION_AID_DATA,
        conversation_text=conversation_text
    )
    observer_rating = call_api(api_base_url, observer_prompt)
    result["Observer_Rating"] = observer_rating
    print(f"  Observer Rating: {observer_rating}")
    
    # Get self rating
    self_prompt = SELF_RATING_PROMPT_TEMPLATE.format(
        decision_aid_data=DECISION_AID_DATA,
        conversation_text=conversation_text
    )
    self_rating = call_api(api_base_url, self_prompt)
    result["Self_Rating"] = self_rating
    print(f"  Self Rating: {self_rating}")
    
    # Get criteria ratings
    for i, criterion in enumerate(CRITERIA):
        criterion_prompt = CRITERIA_PROMPT_TEMPLATE.format(
            decision_aid_data=DECISION_AID_DATA,
            criterion=criterion,
            conversation_text=conversation_text
        )
        rating = call_api(api_base_url, criterion_prompt)
        criterion_name = f"Criterion_{i+1}"
        result[criterion_name] = rating
        print(f"  {criterion_name}: {rating}")
    
    return result


def save_overall_ratings(results: List[Dict], output_file: str):
    """Save overall ratings to CSV, appending to existing file."""
    if not results:
        return
    
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ["Conversation_Id", "User_Rating", "Observer_Rating", "Self_Rating"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for result in results:
            writer.writerow({
                "Conversation_Id": result["Conversation_Id"],
                "User_Rating": result.get("User_Rating", ""),
                "Observer_Rating": result.get("Observer_Rating", ""),
                "Self_Rating": result.get("Self_Rating", ""),
            })


def save_criteria_ratings(results: List[Dict], output_file: str):
    """Save criteria ratings to CSV, appending to existing file."""
    if not results:
        return
    
    file_exists = os.path.exists(output_file)
    
    fieldnames = ["Conversation_Id"] + [f"Criterion_{i+1}" for i in range(len(CRITERIA))]
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for result in results:
            row = {"Conversation_Id": result["Conversation_Id"]}
            for i in range(len(CRITERIA)):
                criterion_name = f"Criterion_{i+1}"
                row[criterion_name] = result.get(criterion_name, "")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Resume evaluation from where it left off")
    parser.add_argument("--port", type=int, default=7471,
                        help="Port where vLLM server is running")
    parser.add_argument("--conv-dir", type=str, default=DEFAULT_CONVERSATIONS_DIR,
                        help="Directory containing conversation CSV files")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save rating CSV files")
    args = parser.parse_args()

    API_BASE_URL = f"http://localhost:{args.port}"

    # Check server
    if not check_server_status(API_BASE_URL):
        print(f"Error: vLLM server not accessible at {API_BASE_URL}")
        print("Start it first")
        return

    print(f"✅ vLLM server is accessible at {API_BASE_URL}")
    
    # Get model name from server
    MODEL_NAME = get_model_name(API_BASE_URL)
    print(f"   Using model: {MODEL_NAME}")

    conversations_dir = args.conv_dir
    output_dir = args.output_dir
    overall_ratings_file = os.path.join(output_dir, "overall_ratings.csv")
    criteria_ratings_file = os.path.join(output_dir, "criteria_ratings.csv")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get list of all conversation files
    conv_files = sorted([
        f for f in os.listdir(conversations_dir)
        if f.startswith("conv_") and f.endswith(".csv")
    ])
    num_convs = len(conv_files)

    # Get already evaluated conversations
    already_done = get_already_evaluated(overall_ratings_file)
    print(f"\nFound {num_convs} total conversations")
    print(f"Already evaluated: {len(already_done)} conversations")
    print(f"Remaining: {num_convs - len(already_done)} conversations")

    # Extract conversation indices and filter out already done
    remaining_indices = [i for i in range(num_convs) if i not in already_done]
    
    if not remaining_indices:
        print("✅ All conversations already evaluated!")
        return
    
    print(f"\nResuming from conversation {remaining_indices[0]}")
    print(f"Output will be saved to {output_dir}")

    batch_size = 10
    for start_idx in range(0, len(remaining_indices), batch_size):
        end_idx = min(start_idx + batch_size, len(remaining_indices))
        batch_indices = remaining_indices[start_idx:end_idx]

        print(f"\n{'='*80}")
        print(f"Processing conversations {batch_indices[0]} to {batch_indices[-1]}")
        print(f"{'='*80}")

        batch_results = []
        for i in batch_indices:
            result = evaluate_conversation(i, conversations_dir, API_BASE_URL)
            if result:
                batch_results.append(result)
                time.sleep(0.5)  # Small delay between requests

        if batch_results:
            save_overall_ratings(batch_results, overall_ratings_file)
            save_criteria_ratings(batch_results, criteria_ratings_file)
            print(f"\n✅ Saved {len(batch_results)} results to CSV files")

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"Overall ratings: {overall_ratings_file}")
    print(f"Criteria ratings: {criteria_ratings_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
