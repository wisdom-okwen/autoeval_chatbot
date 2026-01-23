#!/bin/bash

# Superjudge Comparative Evaluation Runner
# Usage: ./run_superjudge.sh [sample_size] [seed] [output_file]
# Uses GPT-4o as expert judge via OpenAI API

SAMPLE_SIZE=${1:-80}
SEED=${2:-42}
OUTPUT=${3:-superjudge_comparison.csv}

# Load API key from .env if available
ENV_FILE="/playpen-ssd/wokwen/.env"

if [ -f "$ENV_FILE" ]; then
    export PERSONAL_OPENAI_KEY=$(grep PERSONAL_OPENAI_KEY "$ENV_FILE" | cut -d '=' -f2 | tr -d '\r')
    if [ -z "$PERSONAL_OPENAI_KEY" ]; then
        export PERSONAL_OPENAI_KEY=$(grep OPENAI_API_KEY "$ENV_FILE" | cut -d '=' -f2 | tr -d '\r')
    fi
fi

# Verify API key is set
if [ -z "$PERSONAL_OPENAI_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not found!"
    echo "Please set PERSONAL_OPENAI_KEY or OPENAI_API_KEY in .env at $ENV_FILE"
    exit 1
fi

echo "🔍 Superjudge Comparative Evaluation"
echo "  Sample size: ${SAMPLE_SIZE}"
echo "  Random seed: ${SEED}"
echo "  Output: ${OUTPUT}"
echo "  Judge: GPT-4 Turbo"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Run the Python script
python "$SCRIPT_DIR/superjudge_comparison.py" \
    --sample-size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --output "$OUTPUT"

echo ""
echo "✅ Superjudge evaluation complete!"