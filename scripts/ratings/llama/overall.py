#!/usr/bin/env python3
"""
Llama-based conversation rating with Chain-of-Thought reasoning.
Similar to Qwen implementation but using TinyLlama model on GPU 3.
"""

import argparse
import csv
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

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

    def generate_reasoning_and_rating(self, conversation, max_tokens=200, temperature=0.3):
        """Generate reasoning and rating for a conversation"""
        
        # Truncate conversation if too long (keep last N messages)
        if len(conversation) > 10:
            conversation = conversation[-10:]  # Keep last 10 exchanges
        
        # Create shorter prompt for TinyLlama
        conversation_text = "\n".join([f"{msg['role']}: {msg['content'][:200]}..." if len(msg['content']) > 200 else f"{msg['role']}: {msg['content']}" for msg in conversation])
        
        prompt = f"""Rate this conversation 1-10:

{conversation_text}

Quality rating and reason:"""

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
    # Look for "Rating: X/10" pattern
    rating_pattern = r'Rating:\s*(\d+(?:\.\d+)?)/10'
    match = re.search(rating_pattern, reasoning_text, re.IGNORECASE)
    
    if match:
        try:
            rating = float(match.group(1))
            return max(1, min(10, rating))  # Clamp between 1-10
        except ValueError:
            pass
    
    # Fallback: look for standalone numbers
    numbers = re.findall(r'\b([1-9]|10)\b', reasoning_text)
    if numbers:
        try:
            rating = float(numbers[-1])  # Use last number found
            return max(1, min(10, rating))
        except ValueError:
            pass
    
    print(f"Warning: Could not extract rating from: {reasoning_text[:100]}...")
    return 5.0  # Default fallback rating


def get_llama_reasoning_and_rating(conversation, reasoner):
    """Get reasoning and rating from Llama model"""
    try:
        reasoning = reasoner.generate_reasoning_and_rating(conversation)
        rating = extract_rating_from_reasoning(reasoning)
        
        # Clean up reasoning text
        reasoning_clean = reasoning.replace('\n', ' ').strip()
        
        return reasoning_clean, rating
        
    except Exception as e:
        print(f"Error in Llama reasoning: {e}")
        return f"Error generating reasoning: {str(e)}", 5.0


def rate_conversation_with_llama(conversation_id, reasoner):
    """Rate a single conversation using Llama model"""
    
    # Construct conversation file path
    conv_file = f"/playpen-ssd/wokwen/projects/autoeval_chatbot/conversations/conv_{conversation_id:03d}.csv"
    
    if not os.path.exists(conv_file):
        print(f"Conversation file not found: {conv_file}")
        return None
    
    conversation = load_conversation_from_csv(conv_file)
    if not conversation:
        return None
    
    print(f"Rating conversation {conversation_id:03d}...")
    
    reasoning, rating = get_llama_reasoning_and_rating(conversation, reasoner)
    
    result = {
        'conversation_id': conversation_id,
        'overall_rating': rating,
        'reasoning': reasoning
    }
    
    print(f"  Rating: {rating}/10")
    print(f"  Reasoning: {reasoning[:100]}...")
    
    return result


def save_results_to_csv(results, output_file):
    """Save rating results to CSV file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    fieldnames = ['conversation_id', 'overall_rating', 'reasoning']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Rate conversations using Llama with Chain-of-Thought')
    parser.add_argument('--start', type=int, default=0, help='Start conversation ID')
    parser.add_argument('--end', type=int, default=19, help='End conversation ID')
    parser.add_argument('--device', type=str, default="0", help='GPU device (0, 1, 2, 3)')
    parser.add_argument('--model', type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help='Llama model to use')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file')
    
    args = parser.parse_args()
    
    # Set device
    device = f"cuda:{args.device}"
    if not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    
    print(f"Using device: {device}")
    print(f"Using model: {args.model}")
    
    # Initialize reasoner
    reasoner = LlamaReasoner(device=device, model_name=args.model)
    
    # Set output file
    if args.output is None:
        model_name = args.model.split('/')[-1].lower()
        args.output = f"/playpen-ssd/wokwen/projects/autoeval_chatbot/ratings/llama/overall/overall_ratings_{model_name}.csv"
    
    print(f"Processing conversations {args.start} to {args.end}")
    print(f"Output file: {args.output}")
    
    results = []
    
    for conv_id in range(args.start, args.end + 1):
        try:
            result = rate_conversation_with_llama(conv_id, reasoner)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing conversation {conv_id}: {e}")
            continue
    
    if results:
        save_results_to_csv(results, args.output)
        print(f"Successfully processed {len(results)} conversations")
    else:
        print("No conversations were successfully processed")


if __name__ == "__main__":
    main()