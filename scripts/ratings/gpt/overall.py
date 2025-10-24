#!/usr/bin/env python3
"""
Overall conversation rating script.
Generates single overall ratings for entire conversations from self, user, and judge perspectives.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utils import (
    RATINGS_OUTPUT_DIR, get_conversation_files, load_conversation
)

# OpenAI setup
client = OpenAI(api_key=os.getenv("PERSONAL_OPENAI_KEY"))

# Overall Rating Prompts
SELF_ASSESSMENT_PROMPT = """
You are the chatbot who just completed this conversation.

Now reflect on your performance as a helper and communicator.

Rate yourself based on:
- How clearly you explained things
- Whether you stayed on topic
- If your tone was appropriate and supportive
- Whether you answered the user's questions effectively
- How well you adapted to the user's language level
- How concise and digestible your responses were
- Whether you avoided technical jargon or overwhelming information
- Whether you repeated yourself or contradicted earlier answers
- How inclusive and respectful your responses were

Score your overall performance in this conversation **from 1 to 10**.

**Only return a single floating-point number between 1 and 10. No explanation.**
"""

USER_ASSESSMENT_PROMPT = """
You are the user who just had this conversation with the chatbot.

You are asked to rate your **overall satisfaction** with the interaction based on:
- How helpful and useful the chatbot was
- Whether your questions were answered clearly and respectfully
- If the responses were easy for you to read and understand
- Whether you felt supported, not judged, and able to ask anything
- Whether the chatbot made you feel more informed, safer, or more confident
- If anything felt confusing, unhelpful, or off-topic

Rate your satisfaction **on a scale from 1 to 10**, with 10 meaning very satisfied and 1 meaning very dissatisfied.

**Only return a single floating-point number between 1 and 10. No explanation.**
"""

THIRD_PARTY_ASSESSMENT_PROMPT = """
You are a third-party evaluator observing a full conversation between a user and an HIV prevention chatbot.

Your job is to analyze the **quality, clarity, consistency, and helpfulness** of the chatbot's responses.

Evaluate based on:
- Whether the chatbot answered each question clearly, accurately, and directly
- If the responses were free of unnecessary jargon and matched the user's language level
- How consistent the chatbot was when handling similar questions or topics
- Whether the chatbot avoided contradictions or repeating itself
- How well the chatbot adapted to changes in the user's tone or focus
- Whether the overall tone was inclusive, supportive, and aligned with health literacy principles
- How well the conversation flowed and stayed focused on HIV prevention or PrEP

Rate the **overall performance of the chatbot in this conversation** on a scale from 1 to 10.

**Only return a single floating-point number between 1 and 10. No explanation.**
"""

def save_incremental_csv(rating: Dict, output_path: Path, headers: List[str]):
    """Save a single rating to CSV file incrementally."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we need headers
    file_exists = output_path.exists()
    
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        
        # Write header only if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the rating data
        writer.writerow(rating)

def get_gpt_overall_rating(prompt: str, conversation: str) -> str:
    """Get GPT rating for overall conversation."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Here is the full conversation:\n\n{conversation}\n\nPlease give your rating."}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=4,
            temperature=0.2
        )
        rating = response.choices[0].message.content.strip()
        # Try to extract numeric value
        import re
        numbers = re.findall(r'\b\d+\.?\d*\b', rating)
        if numbers:
            return float(numbers[0])
        return 5.5  # Default if parsing fails
    except Exception as e:
        print(f"GPT error: {str(e)}")
        return 5.5

def format_conversation_for_rating(conversation: List[Dict]) -> str:
    """Format conversation for rating prompts."""
    formatted = []
    for turn in conversation:
        user_msg = turn.get('user', '').strip()
        bot_resp = turn.get('bot', '').strip()
        if user_msg and bot_resp:
            formatted.append(f"User: {user_msg}\nChatbot: {bot_resp}")
    return "\n\n".join(formatted)

def rate_conversation_overall(conv_id: str, conversation: List[Dict]) -> Dict:
    """Generate overall ratings for a conversation from all three perspectives."""
    if not conversation:
        return {}
    
    # Format conversation for rating
    formatted_convo = format_conversation_for_rating(conversation)
    
    print(f"Rating overall conversation {conv_id}...")
    
    # Get ratings from each perspective
    user_rating = get_gpt_overall_rating(USER_ASSESSMENT_PROMPT, formatted_convo)
    self_rating = get_gpt_overall_rating(SELF_ASSESSMENT_PROMPT, formatted_convo)
    judge_rating = get_gpt_overall_rating(THIRD_PARTY_ASSESSMENT_PROMPT, formatted_convo)
    
    print(f"  User: {user_rating}")
    print(f"  Self: {self_rating}")
    print(f"  Judge: {judge_rating}")
    
    # Calculate conversation metrics
    total_turns = len(conversation)
    user_messages = [turn.get('user', '') for turn in conversation if turn.get('user', '').strip()]
    bot_responses = [turn.get('bot', '') for turn in conversation if turn.get('bot', '').strip()]
    error_count = sum(1 for turn in conversation if turn.get('has_error', False))
    
    avg_user_length = sum(len(msg) for msg in user_messages) / len(user_messages) if user_messages else 0
    avg_bot_length = sum(len(resp) for resp in bot_responses) / len(bot_responses) if bot_responses else 0
    languages = [turn.get('language', 'unknown') for turn in conversation]
    primary_language = max(set(languages), key=languages.count) if languages else 'unknown'
    error_rate = error_count / total_turns if total_turns > 0 else 0
    
    return {
        "conversation_id": conv_id,
        "user_rating": user_rating,
        "self_rating": self_rating,
        "judge_rating": judge_rating,
        "total_turns": total_turns,
        "avg_user_length": round(avg_user_length, 2),
        "avg_bot_length": round(avg_bot_length, 2),
        "primary_language": primary_language,
        "error_rate": round(error_rate, 3),
        "error_count": error_count
    }

def main():
    parser = argparse.ArgumentParser(description="Generate overall conversation ratings")
    parser.add_argument("--start", type=int, default=0, help="Start conversation index")
    parser.add_argument("--end", type=int, default=500, help="End conversation index")
    
    args = parser.parse_args()
    
    print(f"Generating overall ratings for conversations {args.start:03d} to {args.end:03d}")
    print("Processing incrementally...")
    
    # Set up output directory and file
    output_dir = RATINGS_OUTPUT_DIR / "overall"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "overall_ratings.csv"
    
    # Set up headers
    headers = [
        "conversation_id", "user_rating", "self_rating", "judge_rating",
        "total_turns", "avg_user_length", "avg_bot_length", 
        "primary_language", "error_rate", "error_count"
    ]
    
    # Get conversation files
    conv_files = get_conversation_files(args.start, args.end)
    
    processed_count = 0
    
    for conv_id, conv_file in conv_files:
        conversation = load_conversation(conv_file)
        if not conversation:
            print(f"Skipping empty conversation {conv_id}")
            continue
        
        # Generate overall ratings
        overall_rating = rate_conversation_overall(conv_id, conversation)
        if overall_rating:
            # Save immediately
            save_incremental_csv(overall_rating, output_path, headers)
            processed_count += 1
            print(f"✅ Completed {conv_id} ({processed_count}/{len(conv_files)})")
        else:
            print(f"⚠️  Could not rate conversation {conv_id}")
        
        # Progress update every 25 conversations
        if processed_count % 25 == 0:
            print(f"\n🎯 Progress: {processed_count}/{len(conv_files)} conversations processed")
    
    print(f"\n🎉 All done! Processed {processed_count} conversations")
    print(f"CSV file saved to: {output_path}")

if __name__ == "__main__":
    main()