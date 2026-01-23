# Superjudge Comparative Evaluation System

## Overview

The superjudge system uses **GPT-4o** (expert judge) to perform comparative evaluations of conversations across three different chatbot configurations. Each system received identical Dutch user queries about HIV prevention/PrEP, with results saved to `superjudge_eval/data/`.

### Why GPT-4o as Judge?
- Superior reasoning capabilities compared to Llama-70B for nuanced evaluation
- Better at detecting hallucinations and inconsistencies
- Excellent at comparative analysis and medical fact-checking
- More reliable for sophisticated expert judgment tasks

---

## System Architectures Being Evaluated

| System | Configuration | Strength | Weakness |
|--------|---------------|----------|----------|
| **A (Full)** | Structured prompt + Knowledge base | Balanced accuracy + empathy | Potential instruction-data conflicts |
| **B (Prompt-Only)** | Structured prompt only | Perfect tone & instruction adherence | Cannot verify facts; hallucination risk |
| **C (Data-Only)** | Knowledge base only | Factual accuracy; no hallucination | Cold tone; lacks guidance direction |

### Detailed System Breakdown

#### System A: Full System (Production Approach)
```
Configuration:
  • Dutch language user query
  • Full conversation history
  • Detailed guidance instructions (empathy, accuracy, ethics, cultural sensitivity)
  • Comprehensive PrEP knowledge base

Processing:
  • Maintains conversation history
  • Cross-references facts with knowledge base
  • Applies guidance principles
  • Validates accuracy before generating

Expected Result:
  ✅ Accurate information (verified)
  ✅ Appropriate emotional tone (guided)
  ✅ Cultural sensitivity
  ✅ Practical helpfulness
  ⚠️ Potentially complex if balancing too aggressively
```

#### System B: Prompt-Only (Instruction Without Verification)
```
Configuration:
  • Dutch language user query
  • Full conversation history
  • IDENTICAL guidance instructions as System A
  • NO knowledge base

Processing:
  • Maintains conversation history
  • Follows instructions faithfully
  • Generates from training data (no verification)

Expected Result:
  ✅ Perfect emotional tone (instruction-based)
  ✅ Contextual awareness
  ✅ Consistent approach
  ❌ Factual accuracy unverified (hallucination risk)
  ❌ Confident tone may mask incorrect information
  ❌ Outdated information possible (training data cutoff)
```

#### System C: Data-Only (Facts Without Guidance)
```
Configuration:
  • Dutch language user query
  • Full conversation history
  • MINIMAL explicit guidance
  • Comprehensive PrEP knowledge base (same as System A)

Processing:
  • Maintains conversation history
  • Retrieves knowledge base entries
  • Formats facts with minimal editorial guidance

Expected Result:
  ✅ Factually accurate (verified against KB)
  ✅ No hallucination risk
  ✅ Consistent facts
  ❌ Tone may be cold or clinical
  ❌ Poor conversation flow
  ❌ May ignore user emotional state
  ❌ Limited flexibility when data doesn't apply
```

---

## Evaluation Criteria (Six Dimensions)

1. **Medical Accuracy** - Factual correctness about PrEP (costs, side effects, access in Netherlands, eligibility)
2. **Empathy & Support** - Emotional validation and acknowledgment of user concerns
3. **Cultural Sensitivity** - Respectful, inclusive tone; appropriate for diverse backgrounds
4. **Conversation Flow** - Natural progression; effective use of conversation history
5. **Safety & Ethics** - Privacy protection; appropriate scope of advice; recognition of limits
6. **PrEP-Specific Knowledge** - Depth of prevention information, access resources, support organizations

---

## Usage

### Prerequisites
```bash
# Set OpenAI API key
export OPENAI_API_KEY="sk-..."
```

### Basic Commands

```bash
# Random 20 conversations (default)
./scripts/run_superjudge.sh

# 50 conversations with seed 42
./scripts/run_superjudge.sh 50 42

# Small test (5 conversations)
python scripts/superjudge_comparison.py --sample-size 5 --seed 42

# Specific conversations
python scripts/superjudge_comparison.py --conversations "0,5,10,15,20" --output targeted.csv

# Monitor progress
./scripts/monitor_all.sh
```

### Directory Structure
```
superjudge_eval/
├── README.md                      # This file
├── data/                          # All output files
│   ├── superjudge_comparison.csv
│   └── [other evaluations]
└── run_log.txt                    # Background process logs
```

---

## Output Format

Results CSV files contain:

| Column | Description |
|--------|-------------|
| conversation_id | Which conversation set (0-499) |
| best_system | Winner (A/B/C) |
| rating | Quality rating of best system (1-10) |
| reasoning | Expert analysis (2-3 sentences) |
| num_turns_full | Turns in Full System |
| num_turns_prompt | Turns in Prompt-Only |
| num_turns_data | Turns in Data-Only |

### Response Format

GPT-4o superjudge responds with structured format:
```
BEST SYSTEM: [A/B/C]
RATING: [1-10]
ANALYSIS: [Your 2-3 sentence expert judgment]
```

---

## How GPT-4o Evaluates

### Expert Context Provided
GPT-4o receives detailed information about:

1. **System Architecture** - How each system is configured and what resources it has access to
2. **Response Generation Strategy** - Different approaches each system uses (guidance-driven vs data-driven vs hybrid)
3. **Expected Behavior** - What each system should do well and where limitations exist
4. **Evaluation Framework** - How to interpret behavior in context of system design

### Evaluation Process

```
1. FACT-CHECK PHASE
   → Verify numbers/claims against medical reality
   → Identify hallucinations or implausibilities
   → If major errors: Disqualify System B; favor A or C

2. TONE EVALUATION
   → Is response emotionally appropriate?
   → Cultural sensitivity evident?
   → Does validation occur while being realistic?

3. COMPLETENESS ASSESSMENT
   → Does response address actual question?
   → Important context missing?
   → Actionability present?

4. CONSISTENCY CHECK
   → Do responses align across turns?
   → Contradictions within single turn?
   → Does follow-up confirm or contradict earlier advice?

5. FINAL DECISION
   → Pick best performer
   → Rate 1-10 based on absolute quality
   → Explain reasoning in 2-3 sentences
```

---

## Red Flags for Hallucination (System B Detection)

When GPT-4o evaluates System B, it watches for:

- ✋ Specific numerical claims about PrEP costs (without verification source)
- ✋ References to Dutch organizations/programs not actually current
- ✋ Confident statements conflicting with known Netherlands health system
- ✋ Different facts mentioned across conversation turns
- ✋ Claims about "always free" or "always costs X" (too absolute)

---

## Expected Performance

### Baseline Predictions

- **System A wins**: ~60-70% (balanced approach superior)
- **System B wins**: ~10-20% (when accuracy doesn't matter or tone exceptional)
- **System C wins**: ~10-20% (when accuracy critical or facts sufficient)

### Why System A Usually Wins

1. Chatbots that fail on facts damage trust completely
2. Good tone without accuracy = polished misinformation
3. Facts without warmth = cold but fixable
4. Both together (A) = accurate AND trusted

### Common Scenarios

**Scenario 1: All three good**
- Expected: A wins ~70%
- Reason: Balanced approach superior
- Ratings: A=8-9, B=7-8, C=7-8

**Scenario 2: System B hallucinated facts**
- Expected: A or C wins, B rating <5
- Reason: Accuracy > tone in medical context
- Ratings: A=8, B=3, C=8

**Scenario 3: System C factually perfect but cold**
- Expected: A wins
- Reason: Factual accuracy alone insufficient for support chatbot
- Ratings: A=8, C=6 (both accurate, A has better tone)

**Scenario 4: Question is mostly emotional**
- Expected: B might win despite no data
- Reason: Tone and empathy matter most
- Ratings: B=8-9, A=7-8, C=5-6

---

## Technical Details

### API Configuration
- **Model**: GPT-4o (OpenAI)
- **Temperature**: 0.3 (deterministic, not creative)
- **Max Tokens**: 400 (allows full reasoning)
- **Seed**: Command-line argument (enables reproducibility across runs)

### Processing Times
- **5 conversations**: ~30-60 seconds
- **50 conversations**: ~5-10 minutes
- **500 conversations**: ~50-100 minutes

### Cost Considerations
- Each evaluation = 1 GPT-4o API call
- Estimated cost: ~$0.01-0.02 per conversation depending on response length
- Full 500 conversations: ~$5-10

### Reproducibility
- Same seed + same system state = identical judgments
- Different seed = may reach different (but equally valid) conclusions
- All judgments deterministic (temperature=0.3)

---

## Requirements

- OpenAI API key set in `OPENAI_API_KEY` environment variable
- Python 3.8+
- All conversation files present (500 sets each):
  - `conversations/conv_*.csv` (Full System)
  - `baseline/prompt_no_data/convo/conv_*.csv` (Prompt-Only)
  - `baseline/data_no_prompt/convo/conv_*.csv` (Data-Only)

---

## Example Run

```
$ OPENAI_API_KEY="sk-..." ./scripts/run_superjudge.sh 5 42

Superjudge Comparative Evaluation
  Sample size: 5
  Random seed: 42
  Output: superjudge_eval/data/superjudge_comparison.csv

✅ Using gpt-4o as expert judge

Randomly sampled 5 conversations (seed: 42)
Conversation IDs: [6, 17, 33, 89, 142]

Evaluating conversation set 006
  - Full system: 30 turns
  - Prompt-only: 28 turns
  - Data-only: 26 turns
  ✅ Best system: A (Rating: 8.5/10)

🎯 Progress: 5/5 conversations evaluated

==================================================
SUMMARY STATISTICS
==================================================
Total conversations evaluated: 5

Best system distribution:
  A (Full System): 3 (60.0%)
  B (Prompt-Only): 1 (20.0%)
  C (Data-Only): 1 (20.0%)

Average rating of best systems: 8.3/10

✅ Results saved to: superjudge_eval/data/superjudge_comparison.csv
```

---

## Interpretation Guide

### System Performance Patterns

**If System A consistently wins (>70%)**:
- Full approach (prompt + data) superior to isolated strategies
- Guided knowledge effective for this domain

**If System B wins despite no data (<30%)**:
- Prompting alone surprisingly effective
- Instruction quality is high; hallucination rate is low
- Could consider cost reduction through prompt-only approach

**If System C wins (<30%)**:
- Data quality excellent; users value accuracy over tone
- Instructional guidance may add noise
- Facts speak for themselves in this context

**Mixed results (all ~33%)**:
- Different conversation types benefit from different approaches
- Hybrid strategies ideal
- Domain-specific prompt engineering may help System B

---

## Files

- `scripts/superjudge_comparison.py` - Main evaluation script (GPT-4o powered)
- `scripts/run_superjudge.sh` - Shell wrapper
- `scripts/monitor_all.sh` - Progress monitoring utility
- `superjudge_eval/data/` - All output CSV files
- `README.md` - This comprehensive documentation

---

## Notes

- Each evaluation = 1 GPT-4o API call
- Reasoning column contains full GPT-4o response with structured markers
- All text properly quoted for CSV parsing
- Incremental approach: evaluate one conversation at a time for resilience
- GPT-4o significantly better at detecting hallucinations vs Llama-70B

---

## Future Extensions

- Pairwise comparisons (A vs B, B vs C, A vs C isolation tests)
- Domain segmentation (different user concern types)
- Turn-level analysis (which system wins at different conversation stages)
- Ensemble evaluation (multiple expert judges for confidence scoring)
- Prompt optimization (fine-tune System B prompts to reduce hallucination)
- Failure analysis (deep dive into System C when it performs poorly despite having data)
