import os
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OpenAI_API_KEY = os.getenv('PERSONAL_OPENAI_KEY')
if not OpenAI_API_KEY:
    raise ValueError("PERSONAL_OPENAI_KEY environment variable is not set.")

client = OpenAI(api_key=OpenAI_API_KEY)

DEFAULT_CONVERSATIONS_DIR = "/playpen-ssd/wokwen/projects/autoeval_chatbot/conversations"
DEFAULT_OUTPUT_DIR = "/playpen-ssd/wokwen/projects/autoeval_chatbot/ratings"
DECISION_AID_FILE = "/playpen-ssd/wokwen/projects/shesprepared/decision-aid-content.txt"


def load_decision_aid_data():
    """Load decision aid content for context."""
    try:
        with open(DECISION_AID_FILE, 'r', encoding='utf-8') as f:
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

# Prompts for different perspectives on overall rating
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
    - 1: Unacceptable - Critical or dangerous issues

    Return ONLY a number from 1 to 10."""

OBSERVER_OVERALL_PROMPT_TEMPLATE = """You are a third-party expert evaluator analyzing a conversation between a user and an HIV prevention chatbot.

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

SELF_OVERALL_PROMPT_TEMPLATE = """You are the chatbot reviewing your performance in this conversation.

    You have access to the ShesPrEPared decision aid content (the official guidance you should reference):

    ---DECISION AID CONTENT---
    {decision_aid_data}
    ---END DECISION AID CONTENT---

    Reflect on how well you performed based on:
    - Accuracy and alignment with the decision aid content
    - Clarity and accuracy of your responses
    - How well you understood and answered user questions
    - Whether your tone was supportive and non-judgmental
    - How effectively you provided guidance and resources
    - Whether you maintained consistency throughout
    - If you were transparent about limitations

    Rate your overall performance on a scale from 1 to 10:
    - 10: Exemplary - Exceeded expectations
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

OBSERVER_CRITERION_PROMPT_TEMPLATE = """You are a third-party expert evaluator analyzing a conversation between a user and an HIV prevention chatbot.

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


def format_conversation(csv_rows: List[Dict]) -> str:
    """Format conversation from CSV rows into readable format."""
    formatted = []
    for row in csv_rows:
        user_msg = row.get("user", "").strip()
        bot_msg = row.get("bot", "").strip()
        if user_msg or bot_msg:
            if user_msg:
                formatted.append(f"User: {user_msg}")
            if bot_msg:
                formatted.append(f"Chatbot: {bot_msg}")
    return "\n".join(formatted)


def get_user_overall_rating(conversation_text: str) -> float:
    """Get user perspective overall rating (1-10)."""
    prompt = USER_OVERALL_PROMPT_TEMPLATE.replace('{decision_aid_data}', DECISION_AID_DATA)
    prompt = prompt + f"\n\nCONVERSATION:\n{conversation_text}\n\nProvide your rating."
    
    response_text = call_openai_with_retry(prompt)
    return float(response_text.strip())


def get_observer_overall_rating(conversation_text: str) -> float:
    """Get observer/third-party perspective overall rating (1-10)."""
    prompt = OBSERVER_OVERALL_PROMPT_TEMPLATE.replace('{decision_aid_data}', DECISION_AID_DATA)
    prompt = prompt + f"\n\nCONVERSATION:\n{conversation_text}\n\nProvide your rating."
    
    response_text = call_openai_with_retry(prompt)
    return float(response_text.strip())


def get_self_overall_rating(conversation_text: str) -> float:
    """Get self/chatbot perspective overall rating (1-10)."""
    prompt = SELF_OVERALL_PROMPT_TEMPLATE.replace('{decision_aid_data}', DECISION_AID_DATA)
    prompt = prompt + f"\n\nCONVERSATION:\n{conversation_text}\n\nProvide your rating."
    
    response_text = call_openai_with_retry(prompt)
    return float(response_text.strip())


def get_observer_criterion_rating(criterion: str, conversation_text: str) -> float:
    """Get observer/third-party rating (1-10) for a specific criterion."""
    
    criterion_prompt = OBSERVER_CRITERION_PROMPT_TEMPLATE.replace('{decision_aid_data}', DECISION_AID_DATA).replace('{criterion}', criterion)
    prompt = criterion_prompt + f"\n\nCONVERSATION:\n{conversation_text}\n\nProvide your rating."
    
    response_text = call_openai_with_retry(prompt)
    return float(response_text.strip())


def load_conversation(conv_id: int, conversations_dir: str) -> List[Dict]:
    """Load a single conversation CSV file."""
    conv_file = os.path.join(conversations_dir, f"conv_{conv_id:03d}.csv")
    
    if not os.path.exists(conv_file):
        return []
    
    rows = []
    try:
        with open(conv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"Error loading {conv_file}: {e}")
    
    return rows


def evaluate_conversation(conv_id: int, conversations_dir: str) -> Dict:
    """Evaluate a single conversation and return results."""
    
    # Load conversation
    conv_rows = load_conversation(conv_id, conversations_dir)
    if not conv_rows:
        print(f"Skipping conv_{conv_id:03d}: not found or empty")
        return None
    
    # Format conversation
    conv_text = format_conversation(conv_rows)
    
    # Get three overall ratings sequentially
    user_rating = get_user_overall_rating(conv_text)
    observer_rating = get_observer_overall_rating(conv_text)
    self_rating = get_self_overall_rating(conv_text)
    
    print(f"Conv {conv_id}: User={user_rating}, Observer={observer_rating}, Self={self_rating}")
    
    # Get per-criterion ratings (observer only)
    criterion_ratings = {}
    for criterion in CRITERIA:
        criterion_name = criterion.split(":")[0].strip()
        rating = get_observer_criterion_rating(criterion, conv_text)
        criterion_ratings[criterion_name] = rating
        print(f"  - {criterion_name}: {rating}")
    
    return {
        "conv_id": conv_id,
        "user_rating": user_rating,
        "observer_rating": observer_rating,
        "self_rating": self_rating,
        "criteria": criterion_ratings
    }


def call_openai_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Call OpenAI API with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300
            )
            response_text = response.choices[0].message.content.strip()
            if response_text:
                return response_text
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt 
                print(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    return ""

def save_overall_ratings(results: List[Dict], output_file: str):
    """Save overall ratings to CSV."""
    
    write_headers = not os.path.exists(output_file) or os.stat(output_file).st_size == 0
    
    with open(output_file, mode='a', newline='', encoding='utf-8') as f:
        fieldnames = ["Conversation_Id", "User_Rating", "Observer_Rating", "Self_Rating"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_headers:
            writer.writeheader()
        
        for result in results:
            if result:
                writer.writerow({
                    "Conversation_Id": result["conv_id"],
                    "User_Rating": result["user_rating"],
                    "Observer_Rating": result["observer_rating"],
                    "Self_Rating": result["self_rating"]
                })


def save_criteria_ratings(results: List[Dict], output_file: str):
    """Save per-criterion ratings to CSV."""
    
    write_headers = not os.path.exists(output_file) or os.stat(output_file).st_size == 0
    
    # Get all criterion names
    all_criteria = [c.split(":")[0].strip() for c in CRITERIA]
    
    # Create fieldnames: Conversation_Id, then for each criterion: CriterionName_Rating
    fieldnames = ["Conversation_Id"]
    for criterion in all_criteria:
        fieldnames.append(f"{criterion}_Rating")
    
    with open(output_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_headers:
            writer.writeheader()
        
        for result in results:
            if result:
                row = {"Conversation_Id": result["conv_id"]}
                for criterion_name, rating in result["criteria"].items():
                    row[f"{criterion_name}_Rating"] = rating
                writer.writerow(row)


def main():
    """Main evaluation loop."""
    
    parser = argparse.ArgumentParser(description='Evaluate conversations with LLM')
    parser.add_argument('--conv-dir', type=str, default=DEFAULT_CONVERSATIONS_DIR,
                       help=f'Directory containing conversation CSV files (default: {DEFAULT_CONVERSATIONS_DIR})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Directory to save rating CSV files (default: {DEFAULT_OUTPUT_DIR})')
    args = parser.parse_args()
    
    conversations_dir = args.conv_dir
    output_dir = args.output_dir
    overall_ratings_file = f"{output_dir}/overall_ratings.csv"
    criteria_ratings_file = f"{output_dir}/criteria_ratings.csv"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all conversations
    conv_files = sorted([f for f in os.listdir(conversations_dir) if f.startswith("conv_") and f.endswith(".csv")])
    num_convs = len(conv_files)
    
    print(f"Found {num_convs} conversations in {conversations_dir}")
    print(f"Output will be saved to {output_dir}")
    
    # Evaluate conversations
    batch_size = 10  # Process in batches to manage API rate limits
    for start_idx in range(0, num_convs, batch_size):
        end_idx = min(start_idx + batch_size, num_convs)
        
        print(f"\n{'='*80}")
        print(f"Processing conversations {start_idx} to {end_idx-1}")
        print(f"{'='*80}")
        
        batch_results = []
        for i in range(start_idx, end_idx):
            result = evaluate_conversation(i, conversations_dir)
            if result:
                batch_results.append(result)
        
        # Save batch results
        if batch_results:
            save_overall_ratings(batch_results, overall_ratings_file)
            save_criteria_ratings(batch_results, criteria_ratings_file)
            print(f"Saved {len(batch_results)} results to CSV files")
    
    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"Overall ratings: {overall_ratings_file}")
    print(f"Criteria ratings: {criteria_ratings_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
