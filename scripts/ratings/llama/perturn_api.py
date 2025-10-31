#!/usr/bin/env python3
"""
Llama-70B API-based per-turn conversation rating with Chain-of-Thought reasoning.
Uses SGLang server API endpoint for Meta-Llama-3.1-70B-Instruct model.
"""

import argparse
import csv
import json
import os
import sys
import time
import requests
from pathlib import Path
from typing import Dict, List
import re

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# Configuration
API_BASE_URL = "http://localhost:7471"
MODEL_NAME = "Meta-Llama-3.1-70B-Instruct"

# Directory paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONVERSATIONS_DIR = PROJECT_ROOT / "conversations"
RATINGS_DIR = PROJECT_ROOT / "ratings" / "llama" / "per_turn"

def check_server_status():
    """Check if SGLang server is running and accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def llama_api_call(prompt: str, max_tokens: int = 300, temperature: float = 0.3) -> str:
    """Make API call to SGLang server for text generation"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["\n\n---", "\n\nTurn", "\n\nNext", "\n\nUser:"]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['text'].strip()
        else:
            print(f"Unexpected API response format: {result}")
            return ""
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return ""
    except json.JSONDecodeError as e:
        print(f"Failed to parse API response: {e}")
        return ""

def extract_rating_from_reasoning(reasoning_text: str) -> float:
    """Extract numerical rating from reasoning text"""
    # Look for various rating patterns
    patterns = [
        r'[Rr]ating[:\s]*(\d+(?:\.\d+)?)',
        r'[Ss]core[:\s]*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)/10',
        r'(\d+(?:\.\d+)?)\s*out of 10',
        r'\b([1-9]|10)(?:\.\d+)?\b'  # Numbers 1-10 with optional decimals
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, reasoning_text)
        if matches:
            try:
                rating = float(matches[-1])  # Use last match
                return max(1, min(10, rating))  # Clamp between 1-10
            except ValueError:
                continue
    
    print(f"Warning: Could not extract rating from: {reasoning_text[:100]}...")
    return 5.0  # Default fallback rating

def get_llama_turn_reasoning_and_rating(user_msg: str, bot_response: str) -> tuple[str, float]:
    """Get reasoning and rating from Llama API for a single turn"""
    
    # Create prompt for per-turn evaluation
    prompt = f"""You are evaluating a single turn in a conversation about HIV prevention and PrEP information.

Rate this chatbot response based on:
- Clarity and simplicity of the response
- Relevance, accuracy, and helpfulness of information  
- Whether it avoided technical jargon and complex language
- Appropriateness of tone (supportive, respectful, inclusive)
- Whether it answered the user's specific question without straying off-topic
- Use of short sentences and accessible words

User: {user_msg}

Chatbot: {bot_response}

Provide your reasoning in 2-3 sentences, then end with exactly "Rating: X/10" where X is a number from 1 to 10. You MUST include the rating in this exact format.

Assessment:"""

    try:
        reasoning = llama_api_call(prompt, max_tokens=200, temperature=0.3)
        if not reasoning:
            return "Error: No response from API", 5.0
        
        rating = extract_rating_from_reasoning(reasoning)
        
        # Clean up reasoning text
        reasoning_clean = reasoning.replace('\n', ' ').strip()
        
        return reasoning_clean, rating
        
    except Exception as e:
        print(f"Error in Llama turn reasoning: {e}")
        return f"Error generating reasoning: {str(e)}", 5.0

def load_conversation_from_csv(csv_path: str) -> List[Dict]:
    """Load conversation data from CSV file"""
    conversation_data = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('user') and row.get('bot'):
                    conversation_data.append({
                        'user': row['user'],
                        'bot': row['bot']
                    })
    except Exception as e:
        print(f"Error loading conversation from {csv_path}: {e}")
        return []
    
    return conversation_data

def init_csv(file_path: Path, max_turns: int = 30):
    """Initialize CSV file with headers."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        headers = ["conversation_id"] + [f"turn_{i}" for i in range(1, max_turns + 1)]
        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)

def rate_conversation_turns_with_llama(conversation_id: int, max_turns: int = 30) -> List:
    """Rate all turns in a conversation using Llama API"""
    
    # Construct conversation file path
    conv_file = CONVERSATIONS_DIR / f"conv_{conversation_id:03d}.csv"
    
    if not conv_file.exists():
        print(f"Conversation file not found: {conv_file}")
        return None
    
    # Load conversation from CSV
    conversation_data = load_conversation_from_csv(str(conv_file))
    
    if not conversation_data:
        print(f"No valid turns found in conversation {conversation_id}")
        return None
    
    print(f"Rating {min(len(conversation_data), max_turns)} turns in conversation {conversation_id:03d}...")
    
    ratings = [conversation_id]
    
    for idx, turn in enumerate(conversation_data[:max_turns]):
        user_msg = turn['user'].strip()
        bot_resp = turn['bot'].strip()
        
        if user_msg and bot_resp:
            reasoning, rating = get_llama_turn_reasoning_and_rating(user_msg, bot_resp)
            ratings.append(rating)
            print(f"  Turn {idx + 1}: {rating}/10")
            # Print abbreviated reasoning for monitoring
            if len(reasoning) > 80:
                print(f"    Reasoning: {reasoning[:80]}...")
            else:
                print(f"    Reasoning: {reasoning}")
        else:
            ratings.append("N/A")  # Missing turn
            print(f"  Turn {idx + 1}: N/A (missing data)")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    # Fill remaining turns with N/A if conversation is shorter than max_turns
    while len(ratings) <= max_turns:
        ratings.append("N/A")
    
    return ratings

def save_turn_results_to_csv(results: List[List], output_file: Path, max_turns: int):
    """Save per-turn rating results to CSV file"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    headers = ["conversation_id"] + [f"turn_{i}" for i in range(1, max_turns + 1)]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    
    print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Rate conversation turns using Llama-70B API with Chain-of-Thought')
    parser.add_argument('--start', type=int, default=0, help='Start conversation ID')
    parser.add_argument('--end', type=int, default=19, help='End conversation ID')
    parser.add_argument('--max-turns', type=int, default=30, help='Maximum turns to rate per conversation')
    parser.add_argument('--output', type=str, default="per_turn_ratings.csv", help='Output CSV filename')
    
    args = parser.parse_args()
    
    print(f"Llama-70B Per-Turn Rating Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API Endpoint: {API_BASE_URL}")
    print(f"  Conversation range: {args.start} to {args.end}")
    print(f"  Max turns per conversation: {args.max_turns}")
    print("")
    
    # Check if server is running
    if not check_server_status():
        print("❌ Error: SGLang server is not accessible!")
        print(f"Please ensure the server is running on {API_BASE_URL}")
        print("You can check with: curl http://localhost:7471/health")
        sys.exit(1)
    
    print("✅ SGLang server is accessible")
    
    # Set output file path
    output_file = RATINGS_DIR / args.output
    
    print(f"Processing conversations {args.start} to {args.end}")
    print(f"Output file: {output_file}")
    print("")
    
    results = []
    
    for conv_id in range(args.start, args.end + 1):
        try:
            result = rate_conversation_turns_with_llama(conv_id, args.max_turns)
            if result:
                results.append(result)
                print(f"✅ Completed conversation {conv_id}")
            else:
                print(f"❌ Failed to process conversation {conv_id}")
        except Exception as e:
            print(f"❌ Error processing conversation {conv_id}: {e}")
            continue
        
        # Progress update
        if (conv_id - args.start + 1) % 5 == 0:
            completed = conv_id - args.start + 1
            total = args.end - args.start + 1
            print(f"\n🎯 Progress: {completed}/{total} conversations processed\n")
    
    if results:
        save_turn_results_to_csv(results, output_file, args.max_turns)
        print(f"\n🎉 Successfully processed {len(results)} conversations")
        print(f"Results saved to: {output_file}")
    else:
        print("\n❌ No conversations were successfully processed")

if __name__ == "__main__":
    main()