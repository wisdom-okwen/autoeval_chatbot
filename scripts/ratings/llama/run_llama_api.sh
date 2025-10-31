#!/bin/bash

# Run Llama-70B ratings via SGLang API
# Usage: ./run_llama_api.sh [start_idx] [end_idx]

cd "$(dirname "$0")"

START_IDX=${1:-0}
END_IDX=${2:-4}

echo "Running Llama-70B ratings for conversations ${START_IDX} to ${END_IDX}"
echo "Make sure SGLang server is running on localhost:7471"

python overall_api.py --start $START_IDX --end $END_IDX --output "overall_ratings.csv"

echo "Llama-70B rating completed!"