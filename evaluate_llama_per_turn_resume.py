#!/usr/bin/env python3
"""
Resume per-turn evaluation from where it left off.
Evaluates only conversations that haven't been saved yet.
"""
import os
import re
import csv
import argparse
import requests
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONVERSATIONS_DIR = os.path.join(BASE_DIR, "conversations")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "ratings", "llama")


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
        
        return None
        
    except Exception as e:
        print(f"    ❌ API Error: {e}")
        return None


def load_conversation(conv_file: str) -> list:
    """Load conversation turns from CSV file."""
    turns = []
    try:
        with open(conv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_msg = row.get('user', '').strip()
                bot_msg = row.get('bot', '').strip()
                if user_msg and bot_msg:
                    turns.append({
                        'user': user_msg,
                        'bot': bot_msg
                    })
    except Exception as e:
        print(f"Error loading conversation {conv_file}: {e}")
    
    return turns


def get_already_evaluated(ratings_file: str) -> set:
    """Get the set of conversation IDs already evaluated."""
    if not os.path.exists(ratings_file):
        return set()
    
    already_done = set()
    try:
        with open(ratings_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                conv_id = row.get('Conversation_ID', '')
                if conv_id:
                    already_done.add(int(conv_id))
    except:
        pass
    
    return already_done


def evaluate_conversation(conv_idx: int, conversations_dir: str, api_base_url: str) -> dict:
    """Evaluate a single conversation's turns."""
    conv_file = os.path.join(conversations_dir, f"conv_{conv_idx:03d}.csv")
    
    if not os.path.exists(conv_file):
        return None
    
    turns = load_conversation(conv_file)
    if not turns:
        return None
    
    print(f"[{conv_idx}/500] Evaluating conv_{conv_idx:03d}...")
    
    result = {"Conversation_ID": f"conv_{conv_idx:03d}"}
    
    # Evaluate each turn
    for turn_num, turn in enumerate(turns, 1):
        prompt = f"""You are evaluating a turn in an HIV prevention chatbot conversation.

Here is the turn:
User: {turn['user']}
Bot: {turn['bot']}

Rate how well the bot responded to the user on a scale of 1-10:
- 10: Excellent response, accurate and helpful
- 1: Completely unhelpful or dangerous

Respond ONLY with a number (1-10)."""

        rating = call_api(api_base_url, prompt)
        result[f"Turn_{turn_num}"] = rating
        if rating:
            print(f"  Turn {turn_num}: {rating}/10")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Resume per-turn evaluation from where it left off")
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
        return

    print(f"✅ vLLM server is accessible at {API_BASE_URL}")
    MODEL_NAME = get_model_name(API_BASE_URL)
    print(f"   Using model: {MODEL_NAME}")

    conversations_dir = args.conv_dir
    output_dir = args.output_dir
    per_turn_ratings_file = os.path.join(output_dir, "per_turn_ratings.csv")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get list of all conversation files
    conv_files = sorted([
        f for f in os.listdir(conversations_dir)
        if f.startswith("conv_") and f.endswith(".csv")
    ])
    num_convs = len(conv_files)

    # Get already evaluated conversations
    already_done = get_already_evaluated(per_turn_ratings_file)
    print(f"\nFound {num_convs} total conversations")
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
        result = evaluate_conversation(i, conversations_dir, API_BASE_URL)
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
