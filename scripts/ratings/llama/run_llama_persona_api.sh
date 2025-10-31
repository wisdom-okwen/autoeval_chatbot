#!/bin/bash

# Run Llama-70B persona ratings via SGLang API
# Usage: ./run_llama_persona_api.sh [start_idx] [end_idx] [personas] [criteria]

cd "$(dirname "$0")"

START_IDX=${1:-0}
END_IDX=${2:-499}
PERSONAS=${3:-"all"}
CRITERIA=${4:-"all"}

echo "Running Llama-70B persona ratings for conversations ${START_IDX} to ${END_IDX}"
echo "Personas: ${PERSONAS}"
echo "Criteria: ${CRITERIA}"
echo "Make sure SGLang server is running on localhost:7471"
echo ""

if [ "$PERSONAS" = "all" ]; then
    python persona_api.py --start $START_IDX --end $END_IDX --criteria $CRITERIA
else
    python persona_api.py --start $START_IDX --end $END_IDX --personas $PERSONAS --criteria $CRITERIA
fi

echo ""
echo "Llama-70B persona rating completed!"