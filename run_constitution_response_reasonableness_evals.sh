#!/bin/bash

# Script to run reasonableness evaluations for constitution_and_response
# This evaluates the response with just the constitution (no reasoning shown to judge)

set -e  # Exit on error

# Configuration
EVAL_PROMPT_DIR="simple_reasonable_recommendation_v3"
EVAL_TARGET="constitution_and_response"
EVALUATOR_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

echo "========================================="
echo "Running constitution_and_response reasonableness evaluations"
echo "Evaluator: $EVALUATOR_MODEL"
echo "Prompt: $EVAL_PROMPT_DIR"
echo "========================================="
echo ""

# Define experiments with their configurations
# Format: run_name:inference_prompt_dir:max_iteration
experiments=(
    "now-09_20_201429:later_constitutional_cot_v2:9"
    "later-09_20_201440:now_constitutional_cot_v2:9"
    "safe-09_19_182118:risky_constitutional_cot_v2:17"
    "risky-09_19_182001:safe_constitutional_cot_v2:17"
)

# Function to run evaluation for a single iteration
run_evaluation() {
    local run_name=$1
    local inference_prompt_dir=$2
    local iteration=$3

    echo "--------------------------------------"
    echo "Evaluating: $run_name / $inference_prompt_dir / iteration-$iteration"
    echo "--------------------------------------"

    python motivated_reasoning/evaluation/local/evaluate_inference_outputs_with_local_model.py \
        --run_name "$run_name" \
        --iteration "$iteration" \
        --inference_prompt_dir "$inference_prompt_dir" \
        --eval_prompt_dir "$EVAL_PROMPT_DIR" \
        --eval_target "$EVAL_TARGET" \
        --evaluator_model "$EVALUATOR_MODEL" \
        --max_tokens 512

    echo "✓ Completed: $run_name / iteration-$iteration"
    echo ""
}

# Process each experiment
for exp_config in "${experiments[@]}"; do
    IFS=':' read -r run_name inference_prompt_dir max_iter <<< "$exp_config"

    echo ""
    echo "========================================="
    echo "Processing experiment: $run_name"
    echo "Inference prompt: $inference_prompt_dir"
    echo "Iterations: 0-$max_iter"
    echo "========================================="
    echo ""

    # Run evaluations for all iterations
    for iter in $(seq 0 $max_iter); do
        run_evaluation "$run_name" "$inference_prompt_dir" "$iter"
    done

    echo "✓✓✓ Completed all iterations for $run_name"
    echo ""
done

echo ""
echo "========================================="
echo "ALL EVALUATIONS COMPLETE!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  evaluation_output/<run_name>/<inference_prompt_dir>/constitution_and_response/evaluator-3-8b-instruct/simple_reasonable_recommendation_v3/"
