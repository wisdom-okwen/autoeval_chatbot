"""
Blinded cross-evaluation using vLLM/Qwen: compares chatbot vs 2 baselines.

For each conversation index, loads the same conv from all 3 sources,
randomly assigns them to ConvA / ConvB / ConvC (blinding the evaluator),
sends all 3 to a single LLM call, and collects an overall rating (1-10)
plus a 1-sentence reason for each.

CSV output records both the actual identity and the randomized label
so position bias can be checked post-hoc.
"""

import os
import csv
import json
import time
import random
import argparse
import requests
from typing import Dict, List
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHATBOT_DIR = os.path.join(BASE_DIR, "conversations")
DEFAULT_BASELINE1_DIR = os.path.join(BASE_DIR, "baseline", "data_no_prompt", "convo")
DEFAULT_BASELINE2_DIR = os.path.join(BASE_DIR, "baseline", "prompt_no_data", "convo")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "ratings", "qwen")
DEFAULT_PORT = 7472

API_BASE_URL = None
MODEL_NAME = None


def load_decision_aid_data() -> str:
    decision_aid_path = os.path.join(
        os.path.dirname(BASE_DIR), "shesprepared", "decision-aid-content.txt"
    )
    with open(decision_aid_path, "r", encoding="utf-8") as f:
        return f.read()

DECISION_AID_DATA = load_decision_aid_data()

CROSS_EVAL_PROMPT = """You are an expert evaluator of HIV prevention chatbots.

You will be shown THREE conversations between a user and different chatbot systems.
The conversations are labeled ConvA, ConvB, and ConvC.
You do NOT know which system produced which conversation.

Your task: Evaluate each conversation independently on how well the chatbot
performs as an HIV prevention and PrEP information assistant.

A good chatbot should:
- Provide accurate, evidence-based information about HIV prevention and PrEP
- Align its responses with the official ShesPrEPared decision aid content (provided below)
- Use clear, accessible language appropriate for the user
- Be supportive, empathetic, and non-judgmental
- Appropriately encourage consultation with healthcare providers
- Be transparent about limitations and uncertainties
- Address the user's specific questions and concerns
- Provide culturally sensitive and inclusive responses

---DECISION AID CONTENT (official reference)---
DECISION_AID_PLACEHOLDER
---END DECISION AID CONTENT---

---CONVERSATION A---
CONV_A_PLACEHOLDER
---END CONVERSATION A---

---CONVERSATION B---
CONV_B_PLACEHOLDER
---END CONVERSATION B---

---CONVERSATION C---
CONV_C_PLACEHOLDER
---END CONVERSATION C---

Rate each conversation on a scale from 1 to 10:
- 10: Exemplary - Exceeded expectations
- 9: Excellent - Superior performance
- 8: Very Good - Strong with minor gaps
- 7: Good - Met expectations
- 6: Adequate - Acceptable but with gaps
- 5: Fair - Below expectations
- 4: Poor - Significant issues
- 3: Very Poor - Major issues
- 2: Critical - Serious concerns
- 1: Unacceptable - Critical or dangerous issues

CRITICAL INSTRUCTIONS:
1. Do NOT provide any preamble, explanation, or introduction
2. Do NOT include any text outside the JSON object
3. Output ONLY the JSON object, nothing else
4. Each reason must be exactly one sentence
5. Use exact format below

RESPOND WITH ONLY THIS JSON (no other text):
{"ConvA_Rating": <number>, "ConvA_Reason": "<one sentence>", "ConvB_Rating": <number>, "ConvB_Reason": "<one sentence>", "ConvC_Rating": <number>, "ConvC_Reason": "<one sentence>"}"""


def check_server_status(port: int) -> bool:
    """Check if vLLM server is running."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_model_name(port: int) -> str:
    """Fetch model name from vLLM server."""
    try:
        response = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["id"]
    except Exception as e:
        print(f"Error fetching model name: {e}")
    return None


def format_qwen_prompt(prompt: str) -> str:
    """Format prompt with Qwen chat template."""
    return (
        "<|im_start|>user\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def call_qwen_with_retry(prompt: str, port: int, max_retries: int = 3) -> str:
    """Call vLLM server with exponential backoff retry."""
    formatted = format_qwen_prompt(prompt)
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"http://localhost:{port}/v1/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": MODEL_NAME,
                    "prompt": formatted,
                    "max_tokens": 500,
                    "temperature": 0.2,
                    "stop": ["<|im_end|>", "\n"],
                },
                timeout=120,
            )
            response.raise_for_status()

            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                text = result["choices"][0]["text"].strip()
                if text:
                    return text
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    return ""


def format_conversation(csv_rows: List[Dict]) -> str:
    """Format conversation from CSV rows into readable text."""
    formatted = []
    for row in csv_rows:
        user_msg = row.get("user", "").strip()
        bot_msg = row.get("bot", "").strip()
        if user_msg or bot_msg:
            if user_msg:
                formatted.append(f"User: {user_msg}")
            if bot_msg:
                formatted.append(f"Chatbot: {bot_msg}")
    return "\n".join(formatted)


def load_conversation(conv_id: int, conversations_dir: str) -> List[Dict]:
    """Load a single conversation CSV file."""
    conv_file = os.path.join(conversations_dir, f"conv_{conv_id:03d}.csv")
    if not os.path.exists(conv_file):
        return []
    rows = []
    try:
        with open(conv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"Error loading {conv_file}: {e}")
    return rows


def parse_cross_eval_response(response_text: str) -> Dict:
    """Parse the JSON response from the cross-eval LLM call."""
    text = response_text.strip()

    # Handle markdown code-block wrapping
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip() == "```":
                break
            elif in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    start = text.find("{")
    if start == -1:
        print(f"  [DEBUG] No JSON found. Response: {text[:300]}")
        # Fallback: return dummy values so batch can continue
        return {
            "ConvA_Rating": 5, "ConvA_Reason": "Could not parse response",
            "ConvB_Rating": 5, "ConvB_Reason": "Could not parse response",
            "ConvC_Rating": 5, "ConvC_Reason": "Could not parse response",
        }
    
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as e:
                    print(f"  [DEBUG] JSON parse error: {e}")
                    return {
                        "ConvA_Rating": 5, "ConvA_Reason": "Parse error",
                        "ConvB_Rating": 5, "ConvB_Reason": "Parse error",
                        "ConvC_Rating": 5, "ConvC_Reason": "Parse error",
                    }

    print(f"  [DEBUG] No closing brace found in: {text[:300]}")
    return {
        "ConvA_Rating": 5, "ConvA_Reason": "Malformed response",
        "ConvB_Rating": 5, "ConvB_Reason": "Malformed response",
        "ConvC_Rating": 5, "ConvC_Reason": "Malformed response",
    }


def cross_evaluate_conversation(
    conv_id: int,
    chatbot_dir: str,
    baseline1_dir: str,
    baseline2_dir: str,
    chatbot_label: str,
    baseline1_label: str,
    baseline2_label: str,
    port: int,
) -> Dict:
    """Cross-evaluate one conversation set with randomized blinding."""

    # Load all 3 conversations
    chatbot_rows = load_conversation(conv_id, chatbot_dir)
    baseline1_rows = load_conversation(conv_id, baseline1_dir)
    baseline2_rows = load_conversation(conv_id, baseline2_dir)

    if not chatbot_rows or not baseline1_rows or not baseline2_rows:
        missing = []
        if not chatbot_rows:
            missing.append(chatbot_label)
        if not baseline1_rows:
            missing.append(baseline1_label)
        if not baseline2_rows:
            missing.append(baseline2_label)
        print(f"Skipping conv_{conv_id:03d}: missing from {', '.join(missing)}")
        return None

    # Format
    chatbot_text = format_conversation(chatbot_rows)
    baseline1_text = format_conversation(baseline1_rows)
    baseline2_text = format_conversation(baseline2_rows)

    # Randomize assignment for blinding
    systems = [
        (chatbot_label, chatbot_text),
        (baseline1_label, baseline1_text),
        (baseline2_label, baseline2_text),
    ]
    random.shuffle(systems)

    assignment = {
        "ConvA": systems[0][0],
        "ConvB": systems[1][0],
        "ConvC": systems[2][0],
    }

    # Build prompt (using .replace to avoid curly-brace issues)
    prompt = (
        CROSS_EVAL_PROMPT
        .replace("DECISION_AID_PLACEHOLDER", DECISION_AID_DATA)
        .replace("CONV_A_PLACEHOLDER", systems[0][1])
        .replace("CONV_B_PLACEHOLDER", systems[1][1])
        .replace("CONV_C_PLACEHOLDER", systems[2][1])
    )

    # Single LLM call for all 3
    response_text = call_qwen_with_retry(prompt, port)
    parsed = parse_cross_eval_response(response_text)

    print(
        f"Conv {conv_id:03d}: "
        f"A({assignment['ConvA']})={parsed['ConvA_Rating']}, "
        f"B({assignment['ConvB']})={parsed['ConvB_Rating']}, "
        f"C({assignment['ConvC']})={parsed['ConvC_Rating']}"
    )

    return {
        "conv_id": conv_id,
        "Actual_ConvA": assignment["ConvA"],
        "Actual_ConvB": assignment["ConvB"],
        "Actual_ConvC": assignment["ConvC"],
        "ConvA_Rating": parsed["ConvA_Rating"],
        "ConvA_Reason": parsed["ConvA_Reason"],
        "ConvB_Rating": parsed["ConvB_Rating"],
        "ConvB_Reason": parsed["ConvB_Reason"],
        "ConvC_Rating": parsed["ConvC_Rating"],
        "ConvC_Reason": parsed["ConvC_Reason"],
    }


def save_cross_eval_results(results: List[Dict], output_file: str):
    """Append cross-evaluation results to CSV."""
    write_headers = not os.path.exists(output_file) or os.stat(output_file).st_size == 0

    fieldnames = [
        "Conversation_Id",
        "Actual_ConvA", "Actual_ConvB", "Actual_ConvC",
        "ConvA_Rating", "ConvA_Reason",
        "ConvB_Rating", "ConvB_Reason",
        "ConvC_Rating", "ConvC_Reason",
    ]

    with open(output_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_headers:
            writer.writeheader()
        for result in results:
            if result:
                writer.writerow({
                    "Conversation_Id": result["conv_id"],
                    "Actual_ConvA": result["Actual_ConvA"],
                    "Actual_ConvB": result["Actual_ConvB"],
                    "Actual_ConvC": result["Actual_ConvC"],
                    "ConvA_Rating": result["ConvA_Rating"],
                    "ConvA_Reason": result["ConvA_Reason"],
                    "ConvB_Rating": result["ConvB_Rating"],
                    "ConvB_Reason": result["ConvB_Reason"],
                    "ConvC_Rating": result["ConvC_Rating"],
                    "ConvC_Reason": result["ConvC_Reason"],
                })


def main():
    parser = argparse.ArgumentParser(
        description="Blinded cross-evaluation of chatbot vs baselines using vLLM/Qwen"
    )
    parser.add_argument(
        "--chatbot-dir", type=str, default=DEFAULT_CHATBOT_DIR,
        help="Directory containing chatbot conversation CSVs",
    )
    parser.add_argument(
        "--baseline1-dir", type=str, default=DEFAULT_BASELINE1_DIR,
        help="Directory containing baseline 1 (data_no_prompt) conversation CSVs",
    )
    parser.add_argument(
        "--baseline2-dir", type=str, default=DEFAULT_BASELINE2_DIR,
        help="Directory containing baseline 2 (prompt_no_data) conversation CSVs",
    )
    parser.add_argument(
        "--chatbot-label", type=str, default="chatbot",
        help="Label for the chatbot system (default: chatbot)",
    )
    parser.add_argument(
        "--baseline1-label", type=str, default="data_no_prompt",
        help="Label for baseline 1 (default: data_no_prompt)",
    )
    parser.add_argument(
        "--baseline2-label", type=str, default="prompt_no_data",
        help="Label for baseline 2 (default: prompt_no_data)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save results",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"vLLM server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible shuffling",
    )
    args = parser.parse_args()

    global API_BASE_URL, MODEL_NAME

    # Check server status
    if not check_server_status(args.port):
        print(f"❌ vLLM server not accessible at http://localhost:{args.port}")
        print(f"Please start a vLLM server with Qwen model on port {args.port}")
        exit(1)

    print(f"✅ vLLM server is accessible at http://localhost:{args.port}")

    # Get model name
    MODEL_NAME = get_model_name(args.port)
    if not MODEL_NAME:
        print("❌ Could not fetch model name from vLLM server")
        exit(1)
    print(f"   Using model: {MODEL_NAME}")

    if args.seed is not None:
        random.seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(args.output_dir, "cross_eval_ratings.csv")

    # Find conversations common to all 3 directories
    chatbot_files = set(
        f for f in os.listdir(args.chatbot_dir)
        if f.startswith("conv_") and f.endswith(".csv")
    )
    baseline1_files = set(
        f for f in os.listdir(args.baseline1_dir)
        if f.startswith("conv_") and f.endswith(".csv")
    )
    baseline2_files = set(
        f for f in os.listdir(args.baseline2_dir)
        if f.startswith("conv_") and f.endswith(".csv")
    )

    common_files = sorted(chatbot_files & baseline1_files & baseline2_files)
    conv_ids = [int(f.replace("conv_", "").replace(".csv", "")) for f in common_files]

    print(f"Found {len(conv_ids)} conversations common to all 3 sources")
    print(f"  Chatbot:     {args.chatbot_dir}")
    print(f"  Baseline 1 ({args.baseline1_label}): {args.baseline1_dir}")
    print(f"  Baseline 2 ({args.baseline2_label}): {args.baseline2_dir}")
    print(f"  Output:      {output_file}")
    if args.seed is not None:
        print(f"  Random seed: {args.seed}")

    # Process sequentially in batches (1 API call per conversation)
    batch_size = 10
    for start_idx in range(0, len(conv_ids), batch_size):
        end_idx = min(start_idx + batch_size, len(conv_ids))
        batch_ids = conv_ids[start_idx:end_idx]

        print(f"\n{'='*80}")
        print(f"Processing conversations {batch_ids[0]:03d} to {batch_ids[-1]:03d}")
        print(f"{'='*80}")

        batch_results = []
        for conv_id in batch_ids:
            result = cross_evaluate_conversation(
                conv_id,
                args.chatbot_dir, 
                args.baseline1_dir, 
                args.baseline2_dir,
                args.chatbot_label, 
                args.baseline1_label, 
                args.baseline2_label,
                args.port,
            )
            if result:
                batch_results.append(result)

        if batch_results:
            save_cross_eval_results(batch_results, output_file)
            print(f"Saved {len(batch_results)} results")

    print(f"\n{'='*80}")
    print("Cross-evaluation complete!")
    print(f"Results: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
