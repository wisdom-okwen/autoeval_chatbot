#!/usr/bin/env python3
"""
Qwen-based conversation rating script with Chain-of-Thought reasoning.
Uses local Qwen model to generate detailed explanatory reasoning alongside numerical ratings.
"""

import argparse
import csv
import os
import sys
import torch
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables
load_dotenv()

# Add the parent directory to access utils
sys.path.append(str(Path(__file__).parent.parent))
# Import from gpt utils temporarily, but define our own functions
try:
    from gpt.utils import load_conversation
except ImportError:
    # Fallback if import fails
    pass

def get_conversation_files(start_idx: int, end_idx: int):
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

def load_conversation(conv_path: Path):
    """Load conversation from CSV file."""
    import csv
    conversation = []
    try:
        with open(conv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                conversation.append(row)
    except Exception as e:
        print(f"Error loading {conv_path}: {e}")
        return []
    return conversation

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent.parent
CONVERSATIONS_DIR = BASE_DIR / "conversations"
RATINGS_OUTPUT_DIR = BASE_DIR / "ratings"

# OpenAI setup for numerical ratings (fallback)
client = OpenAI(api_key=os.getenv("PERSONAL_OPENAI_KEY"))

class QwenReasoner:
    """Local Qwen model for generating Chain-of-Thought reasoning."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct", device="auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_env_tokens(self):
        """Load HF tokens from local_models/.env file"""
        env_path = Path('/playpen-ssd/wokwen/local_models/.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        if key in ['HUGGINGFACE_TOKEN_ID', 'HUGGING_FACE_ACCESS_TOKEN']:
                            os.environ['HF_TOKEN'] = value
                            break
    
    def _load_model(self):
        """Load Qwen model and tokenizer."""
        if self.model is not None:
            return
            
        print(f"Loading Qwen model: {self.model_name}")
        
        # Load tokens
        self._load_env_tokens()
        
        # Set cache directory
        os.environ['HF_HOME'] = '/playpen-ssd/wokwen/huggingface_cache'
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        
        print(f"Qwen model loaded on: {next(self.model.parameters()).device}")
    
    def generate_reasoning(self, prompt: str, max_tokens=300) -> str:
        """Generate Chain-of-Thought reasoning for a rating."""
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Format messages for Qwen chat template
        chat_template = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(chat_template, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=True):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
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

# Initialize Qwen reasoner (will be loaded on first use)
qwen_reasoner = None

def get_qwen_reasoner(model_name="Qwen/Qwen2.5-14B-Instruct", device="auto"):
    """Get or initialize the Qwen reasoner."""
    global qwen_reasoner
    if qwen_reasoner is None:
        qwen_reasoner = QwenReasoner(model_name=model_name, device=device)
    return qwen_reasoner

# Chain-of-Thought reasoning prompts (for Qwen) - Direct Objective Format
COT_SELF_REASONING_PROMPT = """
Analyze the chatbot's performance in this HIV prevention and PrEP conversation.

Write 2-3 direct sentences describing the chatbot's clarity of explanations, topic focus, tone appropriateness, question answering effectiveness, language matching, conciseness, jargon avoidance, consistency, and inclusivity. Be factual and objective.

End with: "Rating: X/10"

Conversation:
{conversation}

Assessment:"""

COT_USER_REASONING_PROMPT = """
Evaluate this conversation from the user's perspective who sought HIV prevention and PrEP information.

Write 2-3 direct sentences describing the information helpfulness, response clarity, respectfulness, user support, empowerment level, and any confusion. Be factual and objective.

End with: "Rating: X/10"

Conversation:
{conversation}

Assessment:"""

COT_JUDGE_REASONING_PROMPT = """
Evaluate the chatbot's performance as a third-party expert.

Write 2-3 direct sentences describing the accuracy and directness, language appropriateness, consistency, coherence, adaptability, health literacy alignment, and topic focus. Be factual and objective.

End with: "Rating: X/10"

Conversation:
{conversation}

Assessment:"""

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

def extract_rating_from_reasoning(reasoning: str) -> float:
    """Extract numerical rating from Qwen's reasoning text."""
    import re
    # Look for patterns like "Rating: 8.5" or "I would rate: 7/10" or "Score: 9.0" or just "8.5/10"
    patterns = [
        r'[Rr]ating[:\s]*(\d+\.?\d*)',
        r'[Ss]core[:\s]*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*/\s*10',
        r'(\d+\.?\d*)\s*out of',
        r'rate.*?(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*points?',
        r'\b(\d+\.?\d*)\s*$',  # Number at end of text
        r'Overall:\s*(\d+\.?\d*)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, reasoning)
        if matches:
            try:
                rating = float(matches[-1])  # Take the last match
                if 1 <= rating <= 10:
                    return rating
            except ValueError:
                continue
    
    # If no clear rating found, default to 5.5
    return 5.5

def get_qwen_reasoning_and_rating(prompt_template: str, conversation: str, model_name: str, device: str) -> tuple[str, float]:
    """Get Chain-of-Thought reasoning and extracted rating from Qwen model."""
    try:
        reasoner = get_qwen_reasoner(model_name, device)
        prompt = prompt_template.format(conversation=conversation)
        reasoning = reasoner.generate_reasoning(prompt, max_tokens=200)  # Increased for complete sentences
        
        # Extract numerical rating from the reasoning
        rating = extract_rating_from_reasoning(reasoning)
        
        return reasoning, rating
    except Exception as e:
        print(f"Qwen reasoning error: {str(e)}")
        return "Error generating reasoning - model unavailable.", 5.5

def format_conversation_for_rating(conversation: List[Dict]) -> str:
    """Format conversation for rating prompts."""
    formatted = []
    for turn in conversation:
        user_msg = turn.get('user', '').strip()
        bot_resp = turn.get('bot', '').strip()
        if user_msg and bot_resp:
            formatted.append(f"User: {user_msg}\nChatbot: {bot_resp}")
    return "\n\n".join(formatted)

def rate_conversation_with_qwen(conv_id: str, conversation: List[Dict], model_name: str, device: str) -> Dict:
    """Generate overall ratings with Chain-of-Thought reasoning using Qwen."""
    if not conversation:
        return {}
    
    # Format conversation for rating
    formatted_convo = format_conversation_for_rating(conversation)
    
    print(f"Rating conversation {conv_id} with Qwen CoT reasoning...")
    
    # Get Chain-of-Thought reasoning and ratings from Qwen
    print("  Generating CoT reasoning and ratings...")
    user_reasoning, user_rating = get_qwen_reasoning_and_rating(COT_USER_REASONING_PROMPT, formatted_convo, model_name, device)
    self_reasoning, self_rating = get_qwen_reasoning_and_rating(COT_SELF_REASONING_PROMPT, formatted_convo, model_name, device)
    judge_reasoning, judge_rating = get_qwen_reasoning_and_rating(COT_JUDGE_REASONING_PROMPT, formatted_convo, model_name, device)
    
    print(f"  Qwen ratings - User: {user_rating}, Self: {self_rating}, Judge: {judge_rating}")
    
    # Calculate conversation metrics
    total_turns = len(conversation)
    user_messages = [turn.get('user', '') for turn in conversation if turn.get('user', '').strip()]
    bot_messages = [turn.get('bot', '') for turn in conversation if turn.get('bot', '').strip()]
    
    avg_user_length = sum(len(msg.split()) for msg in user_messages) / len(user_messages) if user_messages else 0
    avg_bot_length = sum(len(msg.split()) for msg in bot_messages) / len(bot_messages) if bot_messages else 0
    
    # Detect primary language (simple heuristic)
    all_text = ' '.join(user_messages + bot_messages).lower()
    if 'de ' in all_text or 'und ' in all_text or 'ich ' in all_text:
        primary_language = 'German'
    elif 'le ' in all_text or 'la ' in all_text or 'je ' in all_text:
        primary_language = 'French'
    elif 'el ' in all_text or 'la ' in all_text or 'yo ' in all_text:
        primary_language = 'Spanish'
    elif 'o ' in all_text or 'a ' in all_text or 'eu ' in all_text:
        primary_language = 'Portuguese'
    else:
        primary_language = 'English'
    
    return {
        'conversation_id': conv_id,
        'user_rating': user_rating,
        'self_rating': self_rating,
        'judge_rating': judge_rating,
        'user_reasoning': user_reasoning,
        'self_reasoning': self_reasoning,
        'judge_reasoning': judge_reasoning,
        'total_turns': total_turns,
        'avg_user_length': round(avg_user_length, 2),
        'avg_bot_length': round(avg_bot_length, 2),
        'primary_language': primary_language,
        'error_rate': 0.0,  # Placeholder
        'error_count': 0    # Placeholder
    }

def main():
    parser = argparse.ArgumentParser(description="Rate conversations with Qwen Chain-of-Thought reasoning")
    parser.add_argument("--start", type=int, default=0, help="Start conversation index")
    parser.add_argument("--end", type=int, default=10, help="End conversation index")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="Qwen model to use")
    parser.add_argument("--device", default="auto", help="Device for Qwen model")
    
    args = parser.parse_args()
    
    print(f"Rating conversations {args.start:03d} to {args.end:03d} with Qwen CoT reasoning")
    print(f"Using Qwen model: {args.model}")
    print(f"Device: {args.device}")
    
    # Convert device format for transformers
    device = args.device
    if device != "auto" and device.isdigit():
        # When CUDA_VISIBLE_DEVICES is set, use cuda:0 (the visible device)
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            device = "cuda:0"
        else:
            device = f"cuda:{device}"
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    # Get conversation files
    conv_files = get_conversation_files(args.start, args.end)
    
    # Set up output in qwen subfolder
    output_dir = RATINGS_OUTPUT_DIR / "qwen" / "overall"
    output_path = output_dir / "overall_ratings.csv"
    
    headers = [
        'conversation_id', 'user_rating', 'self_rating', 'judge_rating',
        'user_reasoning', 'self_reasoning', 'judge_reasoning',
        'total_turns', 'avg_user_length', 'avg_bot_length', 
        'primary_language', 'error_rate', 'error_count'
    ]
    
    processed_count = 0
    
    # Initialize Qwen model once at the start
    print("Initializing Qwen model...")
    get_qwen_reasoner(args.model, device)
    print("✅ Qwen model ready!")
    
    for conv_id, conv_file in conv_files:
        conversation = load_conversation(conv_file)
        if not conversation:
            print(f"Skipping empty conversation {conv_id}")
            continue
        
        # Rate conversation with Qwen CoT
        rating_with_cot = rate_conversation_with_qwen(conv_id, conversation, args.model, device)
        if rating_with_cot:
            # Save immediately after each rating
            save_incremental_csv(rating_with_cot, output_path, headers)
            processed_count += 1
            print(f"✅ Completed conversation {conv_id} ({processed_count}/{len(conv_files)})")
        
        # Progress update
        if processed_count % 5 == 0:
            print(f"\n🎯 Progress: {processed_count}/{len(conv_files)} conversations completed")
    
    print(f"\n🎉 All done! Processed {processed_count} conversations with Qwen CoT reasoning")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()