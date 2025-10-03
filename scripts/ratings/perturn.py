#!/usr/bin/env python3
"""
Per-turn conversation rating script.
Rates each individual turn in conversations using simple approach.
"""

import os
import csv
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utils import (
    RATINGS_OUTPUT_DIR, get_conversation_files, load_conversation
)

# OpenAI setup
client = OpenAI(api_key=os.getenv("PERSONAL_OPENAI_KEY"))

# Per-turn rating prompt (simple version)
THIRD_PARTY_PER_TURN_PROMPT = """
You are a third-party expert reviewing a single user-chatbot interaction from a conversation about HIV prevention and PrEP.

Your goal is to rate how well the chatbot responded to the user's message in this turn.

Evaluate based on:
- Clarity and simplicity of the chatbot's response
- Relevance, accuracy, and helpfulness of the information provided
- Whether the chatbot avoided technical jargon and complex language
- Appropriateness of tone (supportive, respectful, inclusive)
- Whether the chatbot answered the user's specific question without straying off-topic
- If the response used short sentences and accessible words

Rate the quality of this single Q&A turn from 1 to 10.

**Only return a single floating-point number between 1 and 10. No explanation.**
"""

def get_gpt_turn_rating(prompt: str, turn_text: str) -> str:
    """Get GPT rating for a single turn."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Here is the turn to evaluate:\n\n{turn_text}\n\nPlease give your rating."}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=4,
            temperature=0.2
        )
        rating = response.choices[0].message.content.strip()
        return rating
    except Exception as e:
        print(f"GPT error: {str(e)}")
        return "5.0"

def init_csv(file_path: Path, max_turns: int = 30):
    """Initialize CSV file with headers."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        headers = ["conversation_id"] + [f"turn_{i}" for i in range(1, max_turns + 1)]
        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)

def rate_conversation_turns(conv_id: str, conversation: list, max_turns: int = 30) -> list:
    """Rate all turns in a conversation."""
    ratings = [conv_id]
    
    print(f"Rating {min(len(conversation), max_turns)} turns in conversation {conv_id}...")
    
    for idx, turn in enumerate(conversation[:max_turns]):
        user_msg = turn.get('user', '').strip()
        bot_resp = turn.get('bot', '').strip()
        
        if user_msg and bot_resp:
            turn_text = f"User: {user_msg}\nChatbot: {bot_resp}"
            rating = get_gpt_turn_rating(THIRD_PARTY_PER_TURN_PROMPT, turn_text)
            ratings.append(rating)
            print(f"  Turn {idx + 1}: {rating}")
        else:
            ratings.append("N/A")  # Missing turn
            print(f"  Turn {idx + 1}: N/A (missing data)")
    
    # Fill remaining turns with N/A if conversation is shorter than max_turns
    while len(ratings) <= max_turns:
        ratings.append("N/A")
    
    return ratings

def main():
    parser = argparse.ArgumentParser(description="Rate individual turns in conversations")
    parser.add_argument("--start", type=int, default=0, help="Start conversation index")
    parser.add_argument("--end", type=int, default=500, help="End conversation index")
    parser.add_argument("--max-turns", type=int, default=30, help="Maximum turns to rate per conversation")
    
    args = parser.parse_args()
    
    print(f"Rating turns in conversations {args.start:03d} to {args.end:03d}")
    print(f"Max turns per conversation: {args.max_turns}")
    print("Using simple per-turn rating approach...")
    
    # Set up output file
    output_dir = RATINGS_OUTPUT_DIR / "per_turn"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "per_turn_ratings.csv"
    
    # Initialize CSV
    init_csv(output_path, args.max_turns)
    
    # Get conversation files
    conv_files = get_conversation_files(args.start, args.end)
    
    processed_count = 0
    
    for conv_id, conv_file in conv_files:
        conversation = load_conversation(conv_file)
        if not conversation:
            print(f"Skipping empty conversation {conv_id}")
            continue
        
        # Rate all turns in this conversation
        ratings = rate_conversation_turns(conv_id, conversation, args.max_turns)
        
        # Append to CSV immediately
        with open(output_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(ratings)
        
        processed_count += 1
        print(f"✅ Completed conversation {conv_id} ({processed_count}/{len(conv_files)})")
        
        # Progress update every 25 conversations
        if processed_count % 25 == 0:
            print(f"\n🎯 Progress: {processed_count}/{len(conv_files)} conversations processed")
    
    print(f"\n🎉 All done! Processed {processed_count} conversations")
    print(f"CSV file saved to: {output_path}")

if __name__ == "__main__":
    main()