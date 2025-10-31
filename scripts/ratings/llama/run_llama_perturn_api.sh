#!/bin/bash

# Run Llama-70B per-turn ratings via SGLang API
# Usage: ./run_llama_perturn_api.sh [start_idx] [end_idx] [max_turns]

cd "$(dirname "$0")"

START_IDX=${1:-0}
END_IDX=${2:-19}
MAX_TURNS=${3:-30}

echo "Running Llama-70B per-turn ratings for conversations ${START_IDX} to ${END_IDX}"
echo "Max turns per conversation: ${MAX_TURNS}"
echo "Make sure SGLang server is running on localhost:7471"
echo ""

python perturn_api.py --start $START_IDX --end $END_IDX --max-turns $MAX_TURNS --output "per_turn_ratings.csv"

echo ""
echo "Llama-70B per-turn rating completed!"