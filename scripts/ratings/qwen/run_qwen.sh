#!/bin/bash
# Qwen Rating Script Runner
# This script activates the local_models venv and runs the Qwen rating script

set -e

echo "🚀 Starting Qwen rating script..."
echo "Activating local_models venv..."

# Activate the local_models venv
source /playpen-ssd/wokwen/local_models/.venv/bin/activate

# Set environment variables
export HF_HOME=/playpen-ssd/wokwen/huggingface_cache

# Change to the script directory
cd /playpen-ssd/wokwen/projects/autoeval_chatbot/scripts/ratings/qwen

echo "✅ Environment ready!"
echo "Running: python overall.py $@"

# Run the script with all passed arguments
python overall.py "$@"