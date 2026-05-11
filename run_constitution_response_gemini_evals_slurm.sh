#!/bin/bash

# Script to submit SLURM jobs for constitution_and_response reasonableness evaluations with Gemini
# This runs all evaluations in parallel using SLURM

set -e

echo "========================================="
echo "Submitting SLURM jobs for constitution_and_response reasonableness evaluations"
echo "Evaluator: Gemini 2.5 Flash Lite"
echo "Prompt: simple_reasonable_recommendation_v3"
echo "========================================="
echo ""

# Define experiments with their configurations
# Format: run_name:inference_prompt_dir
experiments=(
    "now-09_20_201429:later_constitutional_cot_v2"
    "later-09_20_201440:now_constitutional_cot_v2"
    "safe-09_19_182118:risky_constitutional_cot_v2"
    "risky-09_19_182001:safe_constitutional_cot_v2"
)

EVAL_PROMPT_DIR="simple_reasonable_recommendation_v3"
EVAL_TARGET="constitution_and_response"
EVALUATOR="flash-lite"

# Submit SLURM jobs for each experiment
for exp_config in "${experiments[@]}"; do
    IFS=':' read -r run_name inference_prompt_dir <<< "$exp_config"

    echo ""
    echo "========================================="
    echo "Submitting jobs for: $run_name"
    echo "Inference prompt: $inference_prompt_dir"
    echo "========================================="
    echo ""

    # Submit SLURM jobs for all iterations of this experiment
    motivated_reasoning/evaluation/slurm/evaluate_inference_outputs_with_prompt_gemini_slurm.sh \
        --run_name "$run_name" \
        --inference_prompt_dir "$inference_prompt_dir" \
        --eval_prompt_dir "$EVAL_PROMPT_DIR" \
        --eval_target "$EVAL_TARGET" \
        --evaluator "$EVALUATOR"

    echo "✓ Submitted all iterations for $run_name"
    echo ""
done

echo ""
echo "========================================="
echo "ALL SLURM JOBS SUBMITTED!"
echo "========================================="
echo ""
echo "Check job status with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  slurm_logging/"
echo ""
echo "Results will be saved to:"
echo "  evaluation_output/<run_name>/<inference_prompt_dir>/constitution_and_response/evaluator-gemini-25-flash-lite/simple_reasonable_recommendation_v3/"
