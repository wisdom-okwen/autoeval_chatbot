#!/usr/bin/env python3
"""
Per-turn conversation evaluation using Llama via vLLM API.
Rates each individual turn in conversations.
Format: One row per conversation with Turn_1, Turn_2, ... columns.
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

PER_TURN_PROMPT_TEMPLATE = """You are evaluating a single exchange in a conversation with an HIV prevention chatbot.

CONVERSATION TURN:
User: {user_message}

Bot response: {bot_response}

Rate this exchange from 1-10 based on accuracy, clarity, directness, and helpfulness. Respond with ONLY a number from 1 to 10."""


def check_server_status(api_url: str) -> bool:
    """Check if vLLM API server is running"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_model_name(api_url: str) -> str:
    """Fetch model name from vLLM server"""
    try:
        response = requests.get(f"{api_url}/v1/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["id"]
        return "unknown"
    except Exception as e:
        print(f"Warning: Could not fetch model name: {e}")
        return "unknown"


def call_llama_with_retry(api_url: str, prompt: str, model_name: str, max_retries: int = 3) -> str:
    """Call Llama API with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{api_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 10,
                    "top_p": 0.9,
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Error calling API after {max_retries} attempts: {e}")
                return ""
    return ""


def extract_rating(text: str) -> float:
    """Extract numerical rating from response text"""
    # Look for a number 1-10
    numbers = re.findall(r'\b([1-9]|10)(?:\.\d+)?\b', text)
    if numbers:
        try:
            rating = float(numbers[0])  # Use first match
            return max(1.0, min(10.0, rating))  # Clamp to 1-10
        except ValueError:
            pass
    
    return 5.0  # Default fallback


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


def evaluate_conversation_per_turn(conv_id: str, conversation: List[Dict[str, str]], api_url: str, model_name: str) -> Dict[str, any]:
    """Evaluate each turn in a conversation and return ratings dict"""
    turn_ratings = {"Conversation_ID": conv_id}
    
    for turn_num, turn in enumerate(conversation, 1):
        user_msg = turn.get("user", "").strip()
        bot_resp = turn.get("bot", "").strip()
        
        if not user_msg or not bot_resp:
            continue
        
        prompt = PER_TURN_PROMPT_TEMPLATE.format(
            user_message=user_msg,
            bot_response=bot_resp
        )
        
        response_text = call_llama_with_retry(api_url, prompt, model_name)
        rating = extract_rating(response_text)
        
        turn_ratings[f"Turn_{turn_num}"] = rating
        print(f"  Turn {turn_num}: {rating:.1f}/10")
    
    return turn_ratings


def save_per_turn_ratings(all_ratings: List[Dict], output_dir: str):
    """Save per-turn ratings to CSV in wide format"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "per_turn_ratings.csv")
    
    if not all_ratings:
        print(f"No ratings to save")
        return
    
    try:
        # Get all unique column names
        all_columns = set()
        all_columns.add("Conversation_ID")
        for rating_dict in all_ratings:
            all_columns.update(rating_dict.keys())
        
        # Sort columns: Conversation_ID first, then Turn_1, Turn_2, etc.
        turn_cols = [col for col in all_columns if col.startswith("Turn_")]
        # Sort by turn number
        turn_cols.sort(key=lambda x: int(x.split("_")[1]))
        columns = ["Conversation_ID"] + turn_cols
        
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, restval="")
            writer.writeheader()
            writer.writerows(all_ratings)
        
        print(f"✅ Saved per-turn ratings to {output_file}")
    except Exception as e:
        print(f"Error saving ratings: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate conversations per-turn with Llama via vLLM")
    parser.add_argument("--port", type=int, default=7471,
                        help="vLLM server port (default: 7471)")
    parser.add_argument("--conv-dir", type=str, default=DEFAULT_CONVERSATIONS_DIR,
                        help="Directory containing conversation CSV files")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save rating CSV files")
    args = parser.parse_args()

    API_BASE_URL = f"http://localhost:{args.port}"

    # Check server
    if not check_server_status(API_BASE_URL):
        print(f"Error: vLLM server not accessible at {API_BASE_URL}")
        print("Start it first with: python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.2-3B-Instruct --port 7471")
        return

    print(f"✅ vLLM server is accessible at {API_BASE_URL}")
    
    # Get model name from server
    MODEL_NAME = get_model_name(API_BASE_URL)
    print(f"   Using model: {MODEL_NAME}")

    conversations_dir = args.conv_dir
    output_dir = args.output_dir

    # Get list of conversations
    if not os.path.isdir(conversations_dir):
        print(f"Error: Conversations directory not found: {conversations_dir}")
        return

    csv_files = sorted(Path(conversations_dir).glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {conversations_dir}")
        return

    print(f"Found {len(csv_files)} conversation files")
    
    all_ratings = []
    start_time = time.time()

    for idx, csv_file in enumerate(csv_files, 1):
        conv_id = csv_file.stem
        print(f"\n[{idx}/{len(csv_files)}] Evaluating {conv_id}...")
        
        conversation = load_conversation_csv(str(csv_file))
        if not conversation:
            print(f"  ⚠️  No turns found in {csv_file.name}")
            continue
        
        turn_ratings = evaluate_conversation_per_turn(conv_id, conversation, API_BASE_URL, MODEL_NAME)
        all_ratings.append(turn_ratings)

    # Save results
    print(f"\n" + "="*50)
    save_per_turn_ratings(all_ratings, output_dir)
    
    elapsed = time.time() - start_time
    print(f"✅ Evaluation complete! Processed {len(csv_files)} conversations in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
