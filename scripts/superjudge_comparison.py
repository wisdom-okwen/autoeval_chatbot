#!/usr/bin/env python3
"""
Superjudge Comparative Evaluation System
Randomly samples and compares conversations across three baselines:
1. Full system (conversations/)
2. Prompt-only (baseline/prompt_no_data/convo/)
3. Data-only (baseline/data_no_prompt/convo/)

Uses GPT-4 Turbo (expert judge) via OpenAI API to select the best conversation.
"""

import argparse
import csv
import json
import os
import sys
import random
import requests
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# Configuration
PERSONAL_OPENAI_KEY = os.getenv('PERSONAL_OPENAI_KEY')
MODEL_NAME = "gpt-4-turbo"  # Expert judge model (superior reasoning for comparative analysis)

# Directory paths
PROJECT_ROOT = Path(__file__).parent.parent  # /projects/autoeval_chatbot
FULL_CONVERSATIONS_DIR = PROJECT_ROOT / "conversations"
PROMPT_ONLY_DIR = PROJECT_ROOT / "baseline" / "prompt_no_data" / "convo"
DATA_ONLY_DIR = PROJECT_ROOT / "baseline" / "data_no_prompt" / "convo"
SUPERJUDGE_DIR = PROJECT_ROOT / "superjudge_eval"
SUPERJUDGE_DATA_DIR = SUPERJUDGE_DIR / "data"

# Rating criteria for comparison
EVALUATION_CRITERIA = [
    "medical_accuracy",
    "empathy", 
    "cultural_sensitivity",
    "conversation_flow",
    "safety_ethics",
    "prep_specific"
]

def check_api_credentials():
    """Check if OpenAI API key is available"""
    if not PERSONAL_OPENAI_KEY:
        return False
    return True

def gpt_api_call(prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    """Make API call to OpenAI (GPT-4 Turbo) for expert evaluation"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PERSONAL_OPENAI_KEY}"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content'].strip()
        return ""
            
    except Exception as e:
        print(f"API request failed: {e}")
        return ""

def load_conversation(csv_path: str) -> List[Dict]:
    """Load conversation from CSV file"""
    conversation = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('user') and row.get('bot'):
                    conversation.append({
                        'user': row['user'],
                        'bot': row['bot']
                    })
    except Exception as e:
        print(f"Error loading conversation from {csv_path}: {e}")
        return []
    
    return conversation

def format_conversation_for_comparison(conversation: List[Dict]) -> str:
    """Format conversation for side-by-side comparison"""
    formatted = []
    for i, turn in enumerate(conversation, 1):
        formatted.append(f"Turn {i}:")
        formatted.append(f"User: {turn.get('user', '')}")
        formatted.append(f"Bot: {turn.get('bot', '')}")
        formatted.append("")
    return "\n".join(formatted)

def extract_rating_and_choice(response_text: str) -> Tuple[str, float, Dict[str, float]]:
    """Extract system choice, overall rating, and per-criterion ratings from expert analysis"""
    # Look for explicit "BEST SYSTEM:" marker first
    best_system_match = re.search(r'BEST\s*SYSTEM\s*:\s*([A-C])', response_text, re.IGNORECASE)
    choice = None
    if best_system_match:
        choice = best_system_match.group(1).upper()
    
    # Fallback to other patterns if marker not found
    if not choice:
        choice_patterns = [
            r'[Bb]est\s*(?:system|approach)[:\s]*([A-C])',
            r'[Ss]ystem\s*([A-C])\s*(?:is\s*)?best',
            r'([A-C])\s*is\s*(?:the\s*)?best',
        ]
        for pattern in choice_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                choice = matches[0].upper()
                break
    
    # Look for "RATING:" marker first
    rating = 5.0
    rating_match = re.search(r'RATING\s*:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
    if rating_match:
        try:
            rating = float(rating_match.group(1))
        except ValueError:
            pass
    
    rating = max(1, min(10, rating))  # Clamp between 1-10
    
    # Extract per-criterion ratings
    criteria_ratings = {}
    for criterion in EVALUATION_CRITERIA:
        pattern = f"{criterion.upper()}\\s*:\\s*(\\d+(?:\\.\\d+)?)"
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                criteria_ratings[criterion] = max(1, min(10, score))
            except ValueError:
                criteria_ratings[criterion] = 5.0
        else:
            criteria_ratings[criterion] = 5.0
    
    return choice or "UNKNOWN", rating, criteria_ratings

def run_superjudge_comparison(conv_id: int) -> Dict:
    """Run superjudge comparison for a conversation set"""
    
    # Load conversations from all three sources
    full_conv_path = FULL_CONVERSATIONS_DIR / f"conv_{conv_id:03d}.csv"
    prompt_only_path = PROMPT_ONLY_DIR / f"conv_{conv_id:03d}.csv"
    data_only_path = DATA_ONLY_DIR / f"conv_{conv_id:03d}.csv"
    
    # Check all three exist
    if not full_conv_path.exists():
        print(f"Full conversation not found: {full_conv_path}")
        return None
    
    if not prompt_only_path.exists():
        print(f"Prompt-only conversation not found: {prompt_only_path}")
        return None
        
    if not data_only_path.exists():
        print(f"Data-only conversation not found: {data_only_path}")
        return None
    
    # Load all three conversations
    full_conv = load_conversation(str(full_conv_path))
    prompt_only_conv = load_conversation(str(prompt_only_path))
    data_only_conv = load_conversation(str(data_only_path))
    
    if not all([full_conv, prompt_only_conv, data_only_conv]):
        print(f"Could not load all conversations for conv_id {conv_id:03d}")
        return None
    
    print(f"\nEvaluating conversation set {conv_id:03d}")
    print(f"  - Full system: {len(full_conv)} turns")
    print(f"  - Prompt-only: {len(prompt_only_conv)} turns")
    print(f"  - Data-only: {len(data_only_conv)} turns")
    
    # Create comparison prompt
    full_formatted = format_conversation_for_comparison(full_conv)
    prompt_formatted = format_conversation_for_comparison(prompt_only_conv)
    data_formatted = format_conversation_for_comparison(data_only_conv)
    
    prompt = f"""EXPERT SYSTEM EVALUATOR: You are analyzing three variants of an HIV prevention/PrEP chatbot to determine which approach produces better user support.

==== SYSTEM CONFIGURATION DETAILS ====

All three systems respond to the same user queries about PrEP (in various languages). Understand these critical differences:

SYSTEM A (FULL SYSTEM - Production baseline):
- Configuration: Uses both structured prompt instructions AND comprehensive PrEP knowledge data
- Prompt Strategy: System receives detailed guidance on how to approach medical accuracy, empathy, cultural sensitivity, and ethical communication
- Context Handling: Maintains full conversation history; references previous turns when appropriate; understands full context of user concerns over time
- Response Style: Synthesizes guidance from explicit prompt instructions with factual information from knowledge base; balanced approach between following instructions and providing accurate data
- Knowledge Base: Has access to comprehensive PrEP data covering costs, side effects, access in Netherlands, support resources, cultural considerations
- Data Usage: Actively incorporates relevant facts into responses; corrects misinformation; provides source-appropriate recommendations

SYSTEM B (PROMPT-ONLY - Instruction guidance without data):
- Configuration: Uses only structured prompt instructions WITHOUT knowledge data
- Prompt Strategy: System receives the same detailed guidance on approach, tone, and methodology as System A
- Context Handling: Maintains conversation history; follows instruction patterns consistently; references previous turns
- Response Style: Relies entirely on instruction adherence - responds according to guidance without factual validation against a knowledge base
- Knowledge Base: Has NO access to specific PrEP data or medical facts
- Data Usage: Cannot verify accuracy; may generate plausible-sounding but potentially incorrect information; inconsistent details possible
- Limitation: Instruction quality becomes the entire evaluation determinant

SYSTEM C (DATA-ONLY - Factual information without structured guidance):
- Configuration: Uses comprehensive PrEP knowledge data WITHOUT structured prompt instructions
- Prompt Strategy: System receives minimal explicit guidance; no detailed instructions on tone or approach
- Context Handling: Maintains conversation history but may not leverage it consistently without prompt guidance
- Response Style: Responds with factual information from knowledge base; style varies based on data relevance rather than consistent instruction
- Knowledge Base: Has access to comprehensive PrEP data like System A
- Data Usage: Can reference facts accurately; provides correct information when available; may be rigid or inconsistent in how information is presented
- Limitation: Lacks coherent instruction strategy; responses may be factual but ineffective at addressing user emotional or cultural needs

==== EVALUATION FRAMEWORK ====

The three systems represent three different approaches to chatbot design:
1. System A = Best practice (proper balance of guidance + accuracy)
2. System B = Guidance without verification (risk of hallucination)
3. System C = Facts without direction (may be accurate but inconsistently delivered)

Your evaluation criteria:
1. MEDICAL ACCURACY: Are facts about PrEP (costs, side effects, access, eligibility) correct? Only A and C have knowledge to verify this.
2. CONSISTENCY: Do responses follow a coherent strategy? A and B should be consistent; C may vary.
3. EMPATHY & CULTURAL SENSITIVITY: Does the system acknowledge user concerns respectfully? A and B designed for this; C may be factual but cold.
4. CONVERSATION FLOW: Does the system use conversation history effectively? A should do this best.
5. APPROPRIATE SCOPE: Does the response address what user actually asked? All should attempt this.
6. SAFETY & ETHICS: Does the system avoid overstepping (e.g., giving medical advice inappropriately)? Important across all.

==== EXPERT ANALYSIS TASK ====

Compare the three conversation threads below. Look for:
- Factual accuracy (verify against what you know about PrEP in Netherlands)
- Internal consistency (do follow-up responses align with earlier statements?)
- Tone appropriateness (empathetic yet professional?)
- Practical usefulness (would this help the user?)
- Red flags (misinformation, inappropriate advice, cold responses, repetition)

---

SYSTEM A (Full System - Prompt + Data):
{full_formatted}

---

SYSTEM B (Prompt-Only - Instruction without data):
{prompt_formatted}

---

SYSTEM C (Data-Only - Data without instruction):
{data_formatted}

---

==== YOUR EXPERT JUDGMENT ====

Based on the system configurations explained above and the conversation quality metrics, provide your analysis:

1. Identify which system produced the BETTER overall response quality
2. Rate the best-performing system on each criterion (1-10 scale):
   - medical_accuracy: Factual correctness about PrEP
   - empathy: Emotional validation and support
   - cultural_sensitivity: Respectful, inclusive tone
   - conversation_flow: Natural progression and coherence
   - safety_ethics: Privacy and ethical handling
   - prep_specific: Depth of PrEP information

Format your response EXACTLY as:
BEST SYSTEM: [A/B/C]
RATING: [1-10]
MEDICAL_ACCURACY: [1-10]
EMPATHY: [1-10]
CULTURAL_SENSITIVITY: [1-10]
CONVERSATION_FLOW: [1-10]
SAFETY_ETHICS: [1-10]
PREP_SPECIFIC: [1-10]
ANALYSIS: [Your 2-3 sentence expert judgment]"""

    try:
        response = gpt_api_call(prompt, max_tokens=500, temperature=0.3)
        if not response:
            print(f"No response from API for conversation {conv_id:03d}")
            return None
        
        choice, rating, criteria_ratings = extract_rating_and_choice(response)
        
        result = {
            'conversation_id': conv_id,
            'best_system': choice,
            'rating': rating,
            'reasoning': response.strip(),
        }
        
        # Add per-criterion ratings
        for criterion in EVALUATION_CRITERIA:
            result[criterion] = criteria_ratings.get(criterion, 5.0)
        
        print(f"  ✅ Best system: {choice} (Rating: {rating}/10)")
        
        return result
        
    except Exception as e:
        print(f"Error in superjudge comparison for {conv_id}: {e}")
        return None

def save_superjudge_results(results: List[Dict], output_file: Path):
    """Save superjudge comparison results to CSV"""
    # Ensure output file is in superjudge_eval/data directory
    if not str(output_file).startswith(str(SUPERJUDGE_DATA_DIR)):
        output_file = SUPERJUDGE_DATA_DIR / output_file.name
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        "conversation_id", "best_system", "rating", "reasoning",
        "medical_accuracy", "empathy", "cultural_sensitivity",
        "conversation_flow", "safety_ethics", "prep_specific"
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Superjudge comparative evaluation of conversation baselines")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of conversations to sample (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="superjudge_comparison.csv", help="Output CSV filename")
    parser.add_argument("--conversations", type=str, default=None, help="Comma-separated conversation IDs to evaluate (overrides random sampling)")
    
    args = parser.parse_args()
    
    print(f"Superjudge Comparative Evaluation")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output: {args.output}")
    print("")
    
    # Check API credentials
    if not check_api_credentials():
        print("❌ Error: OpenAI API key not found!")
        print("Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    print(f"✅ Using {MODEL_NAME} as expert judge")
    print("")
    
    # Determine which conversations to evaluate
    if args.conversations:
        conv_ids = [int(cid.strip()) for cid in args.conversations.split(",")]
        print(f"Evaluating specified conversations: {conv_ids}")
    else:
        # Randomly sample conversations
        random.seed(args.seed)
        available_convs = range(500)  # We have 500 conversations
        conv_ids = sorted(random.sample(available_convs, min(args.sample_size, len(available_convs))))
        print(f"Randomly sampled {len(conv_ids)} conversations (seed: {args.seed})")
        print(f"Conversation IDs: {conv_ids}")
    
    print("")
    
    # Run superjudge evaluation on sampled conversations
    results = []
    
    for idx, conv_id in enumerate(conv_ids, 1):
        result = run_superjudge_comparison(conv_id)
        if result:
            results.append(result)
        
        # Progress update
        if idx % 5 == 0:
            print(f"🎯 Progress: {idx}/{len(conv_ids)} conversations evaluated")
    
    # Save results
    if results:
        output_file = SUPERJUDGE_DATA_DIR / args.output
        save_superjudge_results(results, output_file)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        total_evaluated = len(results)
        system_counts = {}
        for result in results:
            system = result['best_system']
            system_counts[system] = system_counts.get(system, 0) + 1
        
        print(f"Total conversations evaluated: {total_evaluated}")
        print(f"\nBest system counts:")
        for system in ['A', 'B', 'C']:
            count = system_counts.get(system, 0)
            pct = (count / total_evaluated * 100) if total_evaluated > 0 else 0
            system_name = {
                'A': 'Full System',
                'B': 'Prompt-Only',
                'C': 'Data-Only'
            }.get(system, 'Unknown')
            print(f"  {system} ({system_name}): {count} ({pct:.1f}%)")
        
        avg_rating = sum(r['rating'] for r in results) / len(results) if results else 0
        print(f"\nAverage rating of best systems: {avg_rating:.2f}/10")
        print(f"\nDetailed results saved to: {output_file}")
    else:
        print("❌ No conversations were successfully evaluated")

if __name__ == "__main__":
    main()