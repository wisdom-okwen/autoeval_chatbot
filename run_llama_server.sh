#!/bin/bash
# Launch vLLM server for Llama-3.2-3B-Instruct evaluation
#
# Usage:
#   ./run_llama_server.sh                  # defaults: local 3B model, port auto-detected
#   ./run_llama_server.sh /path/to/model   # custom model path

# --- Configuration ---
DEFAULT_MODEL="/playpen-ssd/wokwen/huggingface_cache/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
MODEL_PATH=${1:-$DEFAULT_MODEL}
LOGFILE=$(mktemp /tmp/vllm_server.XXXXXX.log)
IDLE_TIMEOUT=600    # Shut down after 10 min inactivity
CHECK_INTERVAL=10

# --- Find available GPU ---
DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd " " -)
available_gpus=()
for gpu in $DEVICES; do
    processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $gpu)
    if [ -z "$processes" ]; then
        available_gpus+=("$gpu")
    fi
done

if [ ${#available_gpus[@]} -eq 0 ]; then
    echo "No available GPUs. All GPUs are currently in use."
    exit 1
fi
echo "Available GPUs: ${available_gpus[@]}"

# Use first available GPU (3B model fits on 1 GPU)
export CUDA_VISIBLE_DEVICES=${available_gpus[0]}
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# --- Find available port ---
PORT=7471
for port in {7471..7590}; do
    if ! lsof -nP -iTCP:$port -sTCP:LISTEN > /dev/null 2>&1; then
        PORT=$port
        break
    fi
done
echo "Using port: $PORT"

# --- Launch vLLM server ---
echo "Starting vLLM server..."
echo "  Model: $MODEL_PATH"
echo "  Port:  $PORT"
echo "  Log:   $LOGFILE"

touch "$LOGFILE"
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --tensor-parallel-size 1 >> "$LOGFILE" 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start (checking health endpoint)..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✅ Server is ready at http://localhost:$PORT"
        echo ""
        echo "To use with evaluate_llama.py or cross_eval_llama.py:"
        echo "  python3 evaluate_llama.py --port $PORT"
        break
    fi
    sleep 5
done

if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "⚠️  Server not responding after 5 minutes. Check log: $LOGFILE"
fi

# --- Monitor and auto-shutdown on idle ---
LAST_MOD=$(stat -c %Y "$LOGFILE")

while kill -0 $SERVER_PID 2>/dev/null; do
    CURRENT_MOD=$(stat -c %Y "$LOGFILE")
    if [ "$CURRENT_MOD" -gt "$LAST_MOD" ]; then
        LAST_MOD=$CURRENT_MOD
    fi

    CURRENT_TIME=$(date +%s)
    INACTIVE_FOR=$(( CURRENT_TIME - LAST_MOD ))

    if [ "$INACTIVE_FOR" -ge "$IDLE_TIMEOUT" ]; then
        echo "Idle timeout reached (${IDLE_TIMEOUT}s). Shutting down server..."
        kill $SERVER_PID
        break
    fi

    sleep $CHECK_INTERVAL
done

wait $SERVER_PID 2>/dev/null
echo "Server shut down."
