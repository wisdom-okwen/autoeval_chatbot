#!/usr/bin/env python3
"""
Llama-70B API-based conversation rating with Chain-of-Thought reasoning.
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

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# Configuration
API_BASE_URL = "http://localhost:7471"
MODEL_NAME = "Meta-Llama-3.1-70B-Instruct"

# Directory paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONVERSATIONS_DIR = PROJECT_ROOT / "conversations"
RATINGS_DIR = PROJECT_ROOT / "ratings" / "llama" / "overall"

def load_conversation_from_csv(csv_path):
    """Load conversation data from CSV file"""
    conversation = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Add user message
                if row['user'].strip():
                    conversation.append({
                        'role': 'user',
                        'content': row['user']
                    })
                
                # Add bot response
                if row['bot'].strip():
                    conversation.append({
                        'role': 'assistant', 
                        'content': row['bot']
                    })
    except FileNotFoundError:
        print(f"Error: Could not find conversation file: {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading conversation: {e}")
        return None
    
    return conversation


def check_server_status():
    """Check if SGLang server is running and accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def llama_api_call(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Make API call to SGLang server for text generation"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["\n\nRating:", "Rating:", "\n\n---"]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
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


def extract_rating_from_text(text: str) -> float:
    """Extract numerical rating from response text"""
    import re
    
    # Clean the text
    text = text.strip()
    
    # First try to find any number (most common case)
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        try:
            rating = float(numbers[0])  # Take the first number found
            if 1 <= rating <= 10:
                return rating
        except ValueError:
            pass
    
    # Look for patterns like "Rating: 8", "8/10", "Score: 7", etc.
    patterns = [
        r'(?:rating|score):\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)/10',
        r'(\d+(?:\.\d+)?)\s*out\s*of\s*10',
        r'rating.*?(\d+(?:\.\d+)?)',
    ]
    
    text_lower = text.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            try:
                rating = float(matches[-1])  # Take the last match
                if 1 <= rating <= 10:
                    return rating
            except ValueError:
                continue
    
    # If no rating found, return 5.0 as default
    print(f"Warning: Could not extract rating from text: '{text[:50]}...', using default 5.0")
    return 5.0


# Simple rating prompts for Llama
COT_HELPFULNESS_PROMPT = """Rate the HELPFULNESS of the chatbot's responses in this conversation on a scale of 1-10.

Consider:
- Does the chatbot provide useful, actionable information?
- Are responses relevant to the user's questions and concerns?
- Does it offer appropriate resources or next steps?

Conversation:
{conversation}

Rating (1-10):"""

COT_CLARITY_PROMPT = """Rate the CLARITY of the chatbot's responses in this conversation on a scale of 1-10.

Consider:
- Are responses easy to understand and well-organized?
- Does the chatbot use appropriate language for the audience?
- Are explanations clear without being overly technical?

Conversation:
{conversation}

Rating (1-10):"""

COT_ACCURACY_PROMPT = """Rate the ACCURACY of the chatbot's responses in this conversation on a scale of 1-10.

Consider:
- Is the medical and health information factually correct?
- Are recommendations appropriate and evidence-based?
- Does the chatbot avoid giving inappropriate medical advice?

Conversation:
{conversation}

Rating (1-10):"""


def format_conversation_for_rating(conversation: List[Dict]) -> str:
    """Format conversation for rating prompts"""
    formatted = []
    for turn in conversation:
        role = "User" if turn['role'] == 'user' else "Chatbot"
        formatted.append(f"{role}: {turn['content']}")
    return "\n\n".join(formatted)


def get_llama_reasoning_and_rating(prompt_template: str, conversation: str) -> tuple[str, float]:
    """Get rating from Llama API"""
    prompt = prompt_template.format(conversation=conversation)
    
    response = llama_api_call(prompt, max_tokens=10, temperature=0.1)
    if not response:
        return "Unable to generate rating due to API error.", 5.0
    
    # Extract rating from the response text
    rating = extract_rating_from_text(response)
    
    # Clean the response text for CSV formatting
    cleaned_response = clean_response_for_csv(response)
    
    return cleaned_response, rating


def clean_response_for_csv(text: str) -> str:
    """Clean response text to be CSV-friendly"""
    if not text:
        return ""
    
    # Remove extra whitespace and newlines
    cleaned = text.strip()
    
    # Replace newlines with spaces
    cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')
    
    # Remove multiple spaces
    cleaned = ' '.join(cleaned.split())
    
    # If it starts with just a number, make it more descriptive
    import re
    if re.match(r'^\d+(?:\.\d+)?\s*$', cleaned):
        return f"Rating: {cleaned}"
    
    # Truncate if too long (keep first 100 characters)
    if len(cleaned) > 100:
        cleaned = cleaned[:97] + "..."
    
    return cleaned


def rate_conversation_with_llama(conv_id: str, conversation: List[Dict]) -> Dict:
    """Rate a single conversation using Llama 70B model"""
    # Format conversation for prompts
    formatted_convo = format_conversation_for_rating(conversation)
    
    # Truncate if too long for model context
    if len(formatted_convo) > 8000:  # Conservative limit for context
        formatted_convo = formatted_convo[:8000] + "\n\n[Conversation truncated...]"
    
    print(f"Rating {conv_id} with Llama-70B...")
    
    # Get ratings for each dimension
    helpfulness_reasoning, helpfulness_rating = get_llama_reasoning_and_rating(
        COT_HELPFULNESS_PROMPT, formatted_convo
    )
    time.sleep(1)  # Brief pause between API calls
    
    clarity_reasoning, clarity_rating = get_llama_reasoning_and_rating(
        COT_CLARITY_PROMPT, formatted_convo
    )
    time.sleep(1)
    
    accuracy_reasoning, accuracy_rating = get_llama_reasoning_and_rating(
        COT_ACCURACY_PROMPT, formatted_convo
    )
    
    # Calculate conversation metrics
    total_turns = len(conversation)
    user_turns = [turn for turn in conversation if turn['role'] == 'user']
    bot_turns = [turn for turn in conversation if turn['role'] == 'assistant']
    
    avg_user_length = sum(len(turn['content']) for turn in user_turns) / len(user_turns) if user_turns else 0
    avg_bot_length = sum(len(turn['content']) for turn in bot_turns) / len(bot_turns) if bot_turns else 0
    
    # Placeholder for sensitive content detection (implement if needed)
    sensitive_score = 0.0
    sensitive_count = 0
    
    # Determine language (placeholder - implement detection if needed)
    language = "English"
    
    return {
        'conversation_id': conv_id,
        'helpfulness': helpfulness_rating,
        'clarity': clarity_rating,
        'accuracy': accuracy_rating,
        'helpfulness_reasoning': helpfulness_reasoning,
        'clarity_reasoning': clarity_reasoning,
        'accuracy_reasoning': accuracy_reasoning,
        'total_turns': total_turns,
        'avg_user_length': avg_user_length,
        'avg_bot_length': avg_bot_length,
        'language': language,
        'sensitive_score': sensitive_score,
        'sensitive_count': sensitive_count
    }


def get_conversation_files(start_idx: int, end_idx: int):
    """Get list of conversation files in range"""
    conv_files = []
    for conv_idx in range(start_idx, end_idx + 1):
        conv_file = CONVERSATIONS_DIR / f"conv_{conv_idx:03d}.csv"
        if conv_file.exists():
            conv_id = f"conv_{conv_idx:03d}"
            conv_files.append((conv_id, conv_file))
        else:
            print(f"Skipping missing {conv_file.name}")
    return conv_files


def main():
    parser = argparse.ArgumentParser(description="Rate conversations using Llama-70B via SGLang API")
    parser.add_argument("--start", type=int, default=0, help="Start conversation index")
    parser.add_argument("--end", type=int, default=4, help="End conversation index")
    parser.add_argument("--output", default="overall_ratings.csv", help="Output CSV file")
    args = parser.parse_args()
    
    print(f"Llama-70B Rating Script")
    print(f"API URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    
    # Check if server is running
    if not check_server_status():
        print(f"Error: SGLang server not accessible at {API_BASE_URL}")
        print("Please ensure the server is running with: ./run_llm.sh pretrained_models/Meta-Llama-3.1-70B-Instruct")
        sys.exit(1)
    
    print("✓ Server is accessible")
    
    # Create output directory
    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RATINGS_DIR / args.output
    
    # Get conversation files
    conv_files = get_conversation_files(args.start, args.end)
    if not conv_files:
        print("No conversation files found in the specified range.")
        return
    
    print(f"Processing {len(conv_files)} conversations...")
    
    # Process conversations
    results = []
    for conv_id, conv_file in conv_files:
        try:
            conversation = load_conversation_from_csv(conv_file)
            if conversation is None:
                continue
            
            rating_result = rate_conversation_with_llama(conv_id, conversation)
            results.append(rating_result)
            
            print(f"✓ {conv_id}: H={rating_result['helpfulness']:.1f}, C={rating_result['clarity']:.1f}, A={rating_result['accuracy']:.1f}")
            
        except Exception as e:
            print(f"Error processing {conv_id}: {e}")
            continue
    
    # Save results
    if results:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'conversation_id', 'helpfulness', 'clarity', 'accuracy',
                'helpfulness_reasoning', 'clarity_reasoning', 'accuracy_reasoning',
                'total_turns', 'avg_user_length', 'avg_bot_length', 'language',
                'sensitive_score', 'sensitive_count'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ Results saved to {output_path}")
        print(f"Processed {len(results)} conversations successfully")
        
        # Show summary statistics
        if results:
            avg_helpfulness = sum(r['helpfulness'] for r in results) / len(results)
            avg_clarity = sum(r['clarity'] for r in results) / len(results)
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
            
            print(f"\nSummary Statistics:")
            print(f"Average Helpfulness: {avg_helpfulness:.2f}")
            print(f"Average Clarity: {avg_clarity:.2f}")
            print(f"Average Accuracy: {avg_accuracy:.2f}")
    else:
        print("No conversations were successfully processed.")


if __name__ == "__main__":
    main()