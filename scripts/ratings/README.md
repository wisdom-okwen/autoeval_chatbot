# Autoeval Chatbot - Rating Scripts Organization

This folder contains rating scripts organized by model type. Each model generates ratings with its own approach:

## Model Types

### `gpt/` - GPT-4o-mini (Numerical Only)
- **Scripts**: `overall.py`, `persona.py`, `perturn.py`, `utils.py`
- **Approach**: Generates numerical ratings (1-10) without explanations
- **Output**: `ratings/gpt/overall/`, `ratings/gpt/persona/`, `ratings/gpt/per_turn/`

### `qwen/` - Qwen with Chain-of-Thought
- **Scripts**: `overall.py` (and future: `persona.py`, `perturn.py`)
- **Approach**: Generates detailed CoT reasoning + extracted numerical ratings
- **Output**: `ratings/qwen/overall/`, `ratings/qwen/persona/`, `ratings/qwen/per_turn/`

### `llama/` - Llama with Chain-of-Thought
- **Scripts**: (Coming soon)
- **Approach**: Similar to Qwen, but using Llama models
- **Output**: `ratings/llama/overall/`, etc.

### `deepseek/` - DeepSeek with Chain-of-Thought  
- **Scripts**: (Coming soon)
- **Approach**: Similar to Qwen, but using DeepSeek models
- **Output**: `ratings/deepseek/overall/`, etc.

## Usage Examples

```bash
# GPT ratings (numerical only)
cd scripts/ratings/gpt
python overall.py --start 0 --end 10

# Qwen ratings (CoT + numerical)
cd scripts/ratings/qwen  
export CUDA_VISIBLE_DEVICES=2
python overall.py --start 0 --end 10 --model Qwen/Qwen2.5-14B-Instruct

# Future: Llama ratings
cd scripts/ratings/llama
export CUDA_VISIBLE_DEVICES=3
python overall.py --start 0 --end 10 --model meta-llama/Llama-3.2-7B-Instruct
```

## Output Structure

```
ratings/
├── gpt/           # GPT-4o-mini numerical ratings
│   ├── overall/
│   ├── persona/
│   └── per_turn/
├── qwen/          # Qwen CoT + numerical ratings  
│   ├── overall/
│   ├── persona/
│   └── per_turn/
├── llama/         # Llama CoT + numerical ratings
└── deepseek/      # DeepSeek CoT + numerical ratings
```

All CoT models generate both detailed reasoning and numerical ratings for comprehensive analysis.