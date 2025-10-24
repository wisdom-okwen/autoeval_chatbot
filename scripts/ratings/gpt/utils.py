#!/usr/bin/env python3
"""
Shared utilities for conversation rating system.
Common functions, constants, and data structures used across all rating modules.
"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
CONVERSATIONS_DIR = BASE_DIR / "conversations"
RATINGS_OUTPUT_DIR = BASE_DIR / "ratings"

# OpenAI setup
client = openai.OpenAI(api_key=os.getenv("PERSONAL_OPENAI_KEY"))

# Rating Criteria (consistent across all personas)
RATING_CRITERIA = [
    "medical_accuracy",
    "empathy", 
    "cultural_sensitivity",
    "conversation_flow",
    "safety_ethics",
    "prep_specific"
]

# Persona types
PERSONAS = ["self", "user", "judge"]

def load_conversation(conv_path: Path) -> List[Dict]:
    """Load conversation from CSV file."""
    conversation = []
    try:
        with conv_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_msg = row.get('user', '').strip()
                bot_resp = row.get('bot', '').strip()
                language = row.get('language', 'English').strip()
                has_error = bool(int(row.get('has_error', 0)))
                conversation.append({
                    'user': user_msg,
                    'bot': bot_resp,
                    'language': language,
                    'has_error': has_error
                })
    except Exception as e:
        print(f"Error loading {conv_path}: {e}")
    return conversation

def format_conversation(conversation: List[Dict]) -> str:
    """Format conversation for rating prompts."""
    formatted = []
    for i, turn in enumerate(conversation, 1):
        error_note = " [contains errors]" if turn.get('has_error', False) else ""
        formatted.append(f"Turn {i}:")
        formatted.append(f"User ({turn.get('language', 'unknown')}){error_note}: {turn.get('user', '')}")
        formatted.append(f"Bot: {turn.get('bot', '')}")
        formatted.append("")
    return "\n".join(formatted)

def format_turn_context(turn_idx: int, turn: Dict) -> str:
    """Format a single turn for rating."""
    error_note = " [contains errors]" if turn.get('has_error', False) else ""
    return f"Turn {turn_idx}:\nUser ({turn.get('language', 'unknown')}){error_note}: {turn.get('user', '')}\nBot: {turn.get('bot', '')}"

def get_gpt_rating(persona: str, criterion: str, conversation_text: str, prompts_dict: Dict) -> float:
    """Get GPT rating for given criteria from specific persona perspective."""
    try:
        if persona not in prompts_dict or criterion not in prompts_dict[persona]["criteria_prompts"]:
            return 5.0
        
        system_prompt = prompts_dict[persona]["system_prompt"]
        user_prompt = prompts_dict[persona]["criteria_prompts"][criterion].format(conversation_context=conversation_text)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract rating (look for number 1-10, including decimals)
        import re
        # Look for numbers between 1-10, including decimals
        numbers = re.findall(r'\b(?:[1-9](?:\.\d+)?|10(?:\.0+)?)\b', response_text)
        if numbers:
            rating = float(numbers[0])
            # Ensure rating is within 1-10 range
            return max(1.0, min(10.0, rating))
        
        # Fallback: look for any decimal number
        decimal_numbers = re.findall(r'\b\d+\.?\d*\b', response_text)
        for char in decimal_numbers:
            try:
                rating = float(char)
                if 1 <= rating <= 10:
                    return rating
            except ValueError:
                continue
                
        return 5.0  # Default if parsing fails
    except Exception as e:
        print(f"Error getting GPT rating for {persona}/{criterion}: {e}")
        return 5.0

def calculate_conversation_metrics(conversation: List[Dict]) -> Dict:
    """Calculate quantitative metrics for conversation."""
    if not conversation:
        return {}
    
    total_turns = len(conversation)
    user_messages = [turn.get('user', '') for turn in conversation]
    bot_responses = [turn.get('bot', '') for turn in conversation]
    languages = [turn.get('language', 'unknown') for turn in conversation]
    error_count = sum(1 for turn in conversation if turn.get('has_error', False))
    
    return {
        "total_turns": total_turns,
        "avg_user_length": sum(len(msg) for msg in user_messages) / len(user_messages) if user_messages else 0,
        "avg_bot_length": sum(len(resp) for resp in bot_responses) / len(bot_responses) if bot_responses else 0,
        "primary_language": max(set(languages), key=languages.count),
        "error_rate": error_count / total_turns,
        "unique_languages": len(set(languages))
    }

def save_ratings_csv(ratings: List[Dict], output_path: Path, headers: List[str] = None):
    """Save ratings to CSV file."""
    if not ratings:
        print(f"No ratings to save to {output_path}")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if headers is None:
        headers = list(ratings[0].keys())
    
    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for rating_data in ratings:
            row = []
            for header in headers:
                row.append(rating_data.get(header, ""))
            writer.writerow(row)
    
    print(f"Saved {len(ratings)} ratings to {output_path}")

def get_conversation_files(start_idx: int, end_idx: int) -> List[Tuple[str, Path]]:
    """Get list of conversation files in range."""
    conv_files = []
    for conv_idx in range(start_idx, end_idx + 1):
        conv_file = CONVERSATIONS_DIR / f"conv_{conv_idx:03d}.csv"
        if conv_file.exists():
            conv_id = f"conv_{conv_idx:03d}"
            conv_files.append((conv_id, conv_file))
        else:
            print(f"Skipping missing {conv_file.name}")
    return conv_files

def create_summary_report(ratings: List[Dict], criteria: List[str], personas: List[str], 
                         output_path: Path, title: str):
    """Create a summary report of ratings."""
    if not ratings:
        return
    
    with output_path.open('w', encoding='utf-8') as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        
        for criterion in criteria:
            # Overall average for this criterion
            if f"avg_{criterion}" in ratings[0]:
                ratings_list = [r[f"avg_{criterion}"] for r in ratings if f"avg_{criterion}" in r]
                if ratings_list:
                    avg_rating = sum(ratings_list) / len(ratings_list)
                    f.write(f"{criterion.replace('_', ' ').title()}: {avg_rating:.2f} avg (n={len(ratings_list)})\n")
                    
                    # Per-persona breakdown if available
                    for persona in personas:
                        persona_key = f"{persona}_{criterion}"
                        if persona_key in ratings[0]:
                            persona_ratings_list = [r[persona_key] for r in ratings if persona_key in r]
                            if persona_ratings_list:
                                persona_avg = sum(persona_ratings_list) / len(persona_ratings_list)
                                f.write(f"  {persona}: {persona_avg:.2f}\n")
                    f.write("\n")
            # Direct criterion rating (for single persona)
            elif criterion in ratings[0]:
                ratings_list = [r[criterion] for r in ratings if criterion in r]
                if ratings_list:
                    avg_rating = sum(ratings_list) / len(ratings_list)
                    f.write(f"{criterion.replace('_', ' ').title()}: {avg_rating:.2f} avg (n={len(ratings_list)})\n")
        
        f.write(f"\nTotal conversations evaluated: {len(ratings)}\n")
    
    print(f"Summary saved to {output_path}")