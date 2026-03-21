#!/usr/bin/env python3
"""
Per-turn conversation evaluation for GPT baseline systems.
Rates each individual turn in conversations.
Format: One row per conversation with Turn_1, Turn_2, ... columns.
"""

import os
import re
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List
from openai import OpenAI

# Initialize OpenAI client - will use OPENAI_API_KEY environment variable
try:
    client = OpenAI()
except Exception as e:
    print(f"❌ Error initializing OpenAI client: {e}")
    print("Make sure OPENAI_API_KEY environment variable is set")
    exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_conversation_csv(csv_path: str) -> List[Dict[str, str]]:
    """Load conversation from CSV file"""
    conversation = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("user") and row.get("bot"):
                    conversation.append(row)
    except Exception as e:
        print(f"Error loading conversation from {csv_path}: {e}")
    return conversation

def get_already_evaluated(ratings_file: str) -> set:
    """Get conversation IDs from existing CSV"""
    already_done = set()
    if not os.path.exists(ratings_file):
        return already_done
    try:
        with open(ratings_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    conv_id = int(row.get('Conversation_ID', ''))
                    already_done.add(conv_id)
                except ValueError:
                    pass
    except Exception as e:
        print(f"Warning: Could not read existing ratings: {e}")
    return already_done

def call_gpt_api(prompt: str, model: str = "gpt-4o-mini", max_retries: int = 3) -> str:
    """Call GPT API with retry logic"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=10,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"❌ API Error: {str(e)}")
                return None
    return None

def extract_rating(text: str) -> float:
    """Extract numerical rating from response text"""
    if text is None:
        return None
    numbers = re.findall(r'\b([1-9]|10)(?:\.\d+)?\b', text)
    if numbers:
        try:
            rating = float(numbers[0])
            return max(1.0, min(10.0, rating))
        except ValueError:
            pass
    return None

def evaluate_conversation_per_turn(conv_id: int, conversation: List[Dict[str, str]]) -> Dict[str, any]:
    """Evaluate each turn in a conversation and return ratings dict"""
    turn_ratings = {"Conversation_ID": conv_id}
    
    for turn_num, turn in enumerate(conversation, 1):
        user_msg = turn.get("user", "").strip()
        bot_response = turn.get("bot", "").strip()
        
        prompt = f"""You are evaluating a single exchange in a conversation with an HIV prevention chatbot.

CONVERSATION TURN:
User: {user_msg}

Bot response: {bot_response}

Rate this exchange from 1-10 based on accuracy, clarity, directness, and helpfulness.
Respond ONLY with a number from 1 to 10."""
        
        rating = call_gpt_api(prompt)
        rating = extract_rating(rating) if rating else None
        
        if rating is not None:
            turn_ratings[f"Turn_{turn_num}"] = rating
            print(f"  Turn {turn_num}: {rating:.1f}/10")
        else:
            print(f"  Turn {turn_num}: Failed to get rating")
            turn_ratings[f"Turn_{turn_num}"] = None
        
        time.sleep(0.5)  # Rate limiting
    
    return turn_ratings

def main():
    parser = argparse.ArgumentParser(description="Evaluate per-turn ratings for GPT baseline systems")
    parser.add_argument("--system", required=True, choices=["data_no_prompt", "prompt_no_data"],
                        help="Baseline system to evaluate")
    parser.add_argument("--conv-dir", type=str, help="Directory containing conversation CSV files")
    parser.add_argument("--output-dir", type=str, help="Directory to save rating CSV files")
    args = parser.parse_args()
    
    # Setup paths
    if args.conv_dir:
        conversations_dir = args.conv_dir
    else:
        conversations_dir = os.path.join(BASE_DIR, "baseline", args.system, "convo")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(BASE_DIR, "ratings", "gpt", "baseline", args.system)
    
    os.makedirs(output_dir, exist_ok=True)
    
    per_turn_ratings_file = os.path.join(output_dir, "per_turn_ratings.csv")
    
    # Get conversation files
    conversation_files = sorted(Path(conversations_dir).glob("conv_*.csv"))
    num_convs = len(conversation_files)
    
    print(f"System: {args.system}")
    print(f"Conversations dir: {conversations_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Found {num_convs} total conversations")
    
    # Get already evaluated
    already_done = get_already_evaluated(per_turn_ratings_file)
    print(f"Already evaluated: {len(already_done)} conversations")
    print(f"Remaining: {num_convs - len(already_done)} conversations")
    
    # Extract conversation indices and filter out already done
    remaining_indices = [i for i in range(num_convs) if i not in already_done]
    
    if not remaining_indices:
        print("✅ All conversations already evaluated!")
        return
    
    print(f"\nResuming from conversation {remaining_indices[0]}")
    print(f"Output will be saved to {per_turn_ratings_file}")
    
    for i in remaining_indices:
        conv_file = conversation_files[i]
        conv_id = int(conv_file.stem.split('_')[1])
        
        print(f"\n[{i}/{num_convs}] Evaluating {conv_file.name}...")
        
        conversation = load_conversation_csv(str(conv_file))
        if not conversation:
            print(f"  Skipping: No conversation data")
            continue
        
        result = evaluate_conversation_per_turn(conv_id, conversation)
        
        if result:
            # Append to CSV
            file_exists = os.path.exists(per_turn_ratings_file)
            
            # Get all turn keys
            all_keys = set(['Conversation_ID'])
            for key in result.keys():
                if key.startswith('Turn_'):
                    all_keys.add(key)
            
            # Sort turn keys numerically
            turn_keys = sorted([k for k in all_keys if k.startswith('Turn_')], 
                             key=lambda x: int(x.split('_')[1]))
            fieldnames = ['Conversation_ID'] + turn_keys
            
            with open(per_turn_ratings_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(result)
    
    print(f"\n{'='*80}")
    print("Per-turn evaluation complete!")
    print(f"Per-turn ratings: {per_turn_ratings_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
