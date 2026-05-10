# Reasonableness Evaluation Workflow

This document describes the workflow for running reasonableness evaluations and generating plots.

## Current Status

### Running Evaluations (In Progress)
The constitution_and_response evaluations are currently running in the background.

**Monitor progress:**
```bash
tail -f constitution_response_evals.log
```

**Check if complete:**
```bash
tail -20 constitution_response_evals.log | grep -i "complete"
```

## After Evaluations Complete

### Step 1: Run Analysis
This combines correctness and reasonableness data for each experiment:

```bash
./run_constitution_response_analysis.sh
```

This will:
- Process all 4 experiments (Now, Later, Safe, Risky)
- Match correctness evaluations with reasonableness evaluations
- Create summary files in `analysis_output/reasonableness/`

### Step 2: Generate Plots
Once analysis is complete, regenerate the plots:

```bash
python motivated_reasoning/plotting/plot_reasonableness_proportions_paper.py \
    --all-experiments \
    --evaluator 3-8b-instruct \
    --reasonableness-version simple_reasonable_recommendation_v3
```

This will create plots comparing:
- **constitution_and_reasoning_and_response** (judge sees CoT reasoning)
- **constitution_and_response** (judge only sees final response)

Output will be saved to:
```
plots/reasonableness_proportions/evaluator-3-8b-instruct/simple_reasonable_recommendation_v3/
├── combined_reasonableness_proportions.png
└── combined_reasonableness_proportions.pdf
```

## Quick Reference

### Individual Evaluation (for testing)
```bash
python motivated_reasoning/evaluation/local/evaluate_inference_outputs_with_local_model.py \
    --run_name now-09_20_201429 \
    --iteration 0 \
    --inference_prompt_dir later_constitutional_cot_v2 \
    --eval_prompt_dir simple_reasonable_recommendation_v3 \
    --eval_target constitution_and_response \
    --evaluator_model meta-llama/Meta-Llama-3-8B-Instruct \
    --max_tokens 512
```

### Individual Analysis
```bash
python motivated_reasoning/visualization/study_reasonableness_differences.py \
    --evaluation-dir now-09_20_201429 \
    --output-dir analysis_output/reasonableness
```

## Expected Timeline

- **Evaluations**: ~2-3 hours for all 56 iterations (depends on GPU availability)
- **Analysis**: ~5-10 minutes total
- **Plot generation**: <1 minute

## Troubleshooting

### Check evaluation progress
```bash
# See which iteration is currently running
tail -20 constitution_response_evals.log

# Count completed iterations
grep "✓ Completed:" constitution_response_evals.log | wc -l
```

### Verify outputs
```bash
# Check if evaluation files were created
ls evaluation_output/now-09_20_201429/later_constitutional_cot_v2/constitution_and_response/evaluator-3-8b-instruct/simple_reasonable_recommendation_v3/iteration-0/
```
