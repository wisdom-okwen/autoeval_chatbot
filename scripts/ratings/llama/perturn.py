#!/usr/bin/env python3
"""
Llama-based per-turn conversation rating with Chain-of-Thought reasoning.
Rates each individual turn in conversations using Llama models.
"""

import argparse
import csv
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def load_conversation_from_csv(csv_path):
    """Load conversation data from CSV file"""
    conversation = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('user') and row.get('bot'):
                    conversation.append({
                        'role': 'user',
                        'content': row['user']
                    })
                    conversation.append({
                        'role': 'assistant',
                        'content': row['bot']
                    })
        return conversation
    except Exception as e:
        print(f"Error loading conversation from {csv_path}: {e}")
        return []

class LlamaReasoner:
    def __init__(self, device="cuda:0", model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize Llama model for reasoning and rating"""
        self.device = device
        self.model_name = model_name
        
        print(f"Loading Llama model: {model_name}")
        print(f"Target device: {device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        print(f"Model loaded on: {next(self.model.parameters()).device}")
        print(f"Memory used: {torch.cuda.memory_allocated(device) / 1024**3:.1f} GB")

    def generate_turn_reasoning_and_rating(self, user_msg, bot_response, max_tokens=150, temperature=0.3):
        """Generate reasoning and rating for a single turn"""
        
        # Create shorter prompt for TinyLlama focusing on single turn
        prompt = f"""Rate this chatbot response (1-10):

User: {user_msg[:300]}
Bot: {bot_response[:300]}

Rate clarity, helpfulness, and appropriateness. Give rating and brief reason:"""

        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        
        try:
            chat_template = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except:
            # Fallback for models without chat template
            chat_template = prompt
        
        inputs = self.tokenizer(chat_template, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()

def extract_rating_from_reasoning(reasoning_text):
    """Extract numerical rating from reasoning text"""
    # Look for various rating patterns
    patterns = [
        r'[Rr]ating[:\s]*(\d+(?:\.\d+)?)',
        r'[Ss]core[:\s]*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)/10',
        r'(\d+(?:\.\d+)?)\s*out of 10',
        r'\b([1-9]|10)\b'  # Single digit 1-10
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

def get_llama_turn_reasoning_and_rating(user_msg, bot_response, reasoner):
    """Get reasoning and rating from Llama model for a single turn"""
    try:
        reasoning = reasoner.generate_turn_reasoning_and_rating(user_msg, bot_response)
        rating = extract_rating_from_reasoning(reasoning)
        
        # Clean up reasoning text
        reasoning_clean = reasoning.replace('\n', ' ').strip()
        
        return reasoning_clean, rating
        
    except Exception as e:
        print(f"Error in Llama turn reasoning: {e}")
        return f"Error generating reasoning: {str(e)}", 5.0

def init_csv(file_path: Path, max_turns: int = 30):
    """Initialize CSV file with headers."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        headers = ["conversation_id"] + [f"turn_{i}" for i in range(1, max_turns + 1)]
        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)

def rate_conversation_turns_with_llama(conversation_id, reasoner, max_turns=30):
    """Rate all turns in a conversation using Llama model"""
    
    # Construct conversation file path
    conv_file = f"/playpen-ssd/wokwen/projects/autoeval_chatbot/conversations/conv_{conversation_id:03d}.csv"
    
    if not os.path.exists(conv_file):
        print(f"Conversation file not found: {conv_file}")
        return None
    
    # Load conversation from CSV
    conversation_data = []
    try:
        with open(conv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('user') and row.get('bot'):
                    conversation_data.append({
                        'user': row['user'],
                        'bot': row['bot']
                    })
    except Exception as e:
        print(f"Error loading conversation {conversation_id}: {e}")
        return None
    
    if not conversation_data:
        print(f"No valid turns found in conversation {conversation_id}")
        return None
    
    print(f"Rating {min(len(conversation_data), max_turns)} turns in conversation {conversation_id:03d}...")
    
    ratings = [conversation_id]
    
    for idx, turn in enumerate(conversation_data[:max_turns]):
        user_msg = turn['user'].strip()
        bot_resp = turn['bot'].strip()
        
        if user_msg and bot_resp:
            reasoning, rating = get_llama_turn_reasoning_and_rating(user_msg, bot_resp, reasoner)
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
    
    # Fill remaining turns with N/A if conversation is shorter than max_turns
    while len(ratings) <= max_turns:
        ratings.append("N/A")
    
    return ratings

def save_turn_results_to_csv(results, output_file, max_turns):
    """Save per-turn rating results to CSV file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    headers = ["conversation_id"] + [f"turn_{i}" for i in range(1, max_turns + 1)]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Rate conversation turns using Llama with Chain-of-Thought')
    parser.add_argument('--start', type=int, default=0, help='Start conversation ID')
    parser.add_argument('--end', type=int, default=19, help='End conversation ID')
    parser.add_argument('--device', type=str, default="0", help='GPU device (0, 1, 2, 3)')
    parser.add_argument('--model', type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help='Llama model to use')
    parser.add_argument('--max-turns', type=int, default=30, help='Maximum turns to rate per conversation')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file')
    
    args = parser.parse_args()
    
    # Set device
    device = f"cuda:{args.device}"
    if not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    
    print(f"Using device: {device}")
    print(f"Using model: {args.model}")
    print(f"Max turns per conversation: {args.max_turns}")
    
    # Initialize reasoner
    reasoner = LlamaReasoner(device=device, model_name=args.model)
    
    # Set output file
    if args.output is None:
        model_name = args.model.split('/')[-1].lower()
        args.output = f"/playpen-ssd/wokwen/projects/autoeval_chatbot/ratings/llama/per_turn/per_turn_ratings_{model_name}.csv"
    
    print(f"Processing conversations {args.start} to {args.end}")
    print(f"Output file: {args.output}")
    
    results = []
    
    for conv_id in range(args.start, args.end + 1):
        try:
            result = rate_conversation_turns_with_llama(conv_id, reasoner, args.max_turns)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing conversation {conv_id}: {e}")
            continue
    
    if results:
        save_turn_results_to_csv(results, args.output, args.max_turns)
        print(f"Successfully processed {len(results)} conversations")
    else:
        print("No conversations were successfully processed")

if __name__ == "__main__":
    main()