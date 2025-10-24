#!/bin/bash

# Llama rating script wrapper - runs in local_models virtual environment
# Usage: ./run_llama.sh --start 0 --end 19 --device 3 [--model model_name]

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to local_models directory  
LOCAL_MODELS_DIR="/playpen-ssd/wokwen/local_models"

# Activate the virtual environment
source "$LOCAL_MODELS_DIR/.venv/bin/activate"

# Set HuggingFace cache directory
export HF_HOME="/playpen-ssd/wokwen/huggingface_cache"

# Load environment variables from .env file
if [ -f "$LOCAL_MODELS_DIR/.env" ]; then
    export $(grep -v '^#' "$LOCAL_MODELS_DIR/.env" | xargs)
    # Map tokens to HF_TOKEN for consistency (try different token names)
    if [ ! -z "$LLAMA_HF_TOKEN" ]; then
        export HF_TOKEN="$LLAMA_HF_TOKEN"
    elif [ ! -z "$HUGGING_FACE_ACCESS_TOKEN" ]; then
        export HF_TOKEN="$HUGGING_FACE_ACCESS_TOKEN"
    elif [ ! -z "$HUGGINGFACE_TOKEN_ID" ]; then
        export HF_TOKEN="$HUGGINGFACE_TOKEN_ID"
    fi
fi

# Default values
DEFAULT_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_DEVICE="3"
DEFAULT_START="0"
DEFAULT_END="19"

# Parse command line arguments
START="$DEFAULT_START"
END="$DEFAULT_END"
DEVICE="$DEFAULT_DEVICE"
MODEL="$DEFAULT_MODEL"

while [[ $# -gt 0 ]]; do
    case $1 in
        --start)
            START="$2"
            shift 2
            ;;
        --end)
            END="$2"  
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--start N] [--end N] [--device N] [--model MODEL_NAME]"
            exit 1
            ;;
    esac
done

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES="$DEVICE"

echo "Llama Rating Configuration:"
echo "  Model: $MODEL"
echo "  Device: GPU $DEVICE (mapped to cuda:0)"
echo "  Conversation range: $START to $END"
echo "  HF Cache: $HF_HOME"
echo ""

# Run the Python script
cd "$SCRIPT_DIR"
python overall.py --start "$START" --end "$END" --device "0" --model "$MODEL"