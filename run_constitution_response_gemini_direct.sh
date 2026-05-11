#!/bin/bash

# Script to directly submit SLURM jobs for constitution_and_response evaluations with Gemini
# This bypasses the wrapper script to avoid argument passing issues

set -e

echo "========================================="
echo "Directly submitting Gemini evaluation jobs"
echo "Evaluator: Gemini 2.5 Flash Lite"
echo "Eval Target: constitution_and_response"
echo "Prompt: simple_reasonable_recommendation_v3"
echo "========================================="
echo ""

# Define experiments
experiments=(
    "now-09_20_201429:later_constitutional_cot_v2"
    "later-09_20_201440:now_constitutional_cot_v2"
    "safe-09_19_182118:risky_constitutional_cot_v2"
    "risky-09_19_182001:safe_constitutional_cot_v2"
)

EVAL_PROMPT_DIR="simple_reasonable_recommendation_v3"
EVAL_TARGET="constitution_and_response"
EVALUATOR="flash-lite"
SCRIPT_PATH="motivated_reasoning/evaluation/local/evaluate_inference_outputs_with_prompt_gemini.py"

# SLURM configuration
SLURM_CONFIG="--cpus-per-task=2 --mem=8G --time=1:00:00"

# Process each experiment
for exp_config in "${experiments[@]}"; do
    IFS=':' read -r run_name inference_prompt_dir <<< "$exp_config"

    echo ""
    echo "Processing: $run_name"

    # Find all iterations
    INFERENCE_PATH="inference_output/$run_name/$inference_prompt_dir"
    if [ ! -d "$INFERENCE_PATH" ]; then
        echo "Error: $INFERENCE_PATH does not exist"
        continue
    fi

    ITERATION_DIRS=($(find "$INFERENCE_PATH" -maxdepth 1 -type d -name "iteration-*" | sort -V))
    if [ ${#ITERATION_DIRS[@]} -eq 0 ]; then
        echo "Error: No iteration directories found in $INFERENCE_PATH"
        continue
    fi

    # Extract iteration identifiers
    ITERATIONS=()
    for dir in "${ITERATION_DIRS[@]}"; do
        dirname=$(basename "$dir")
        iteration=$(echo "$dirname" | sed -n 's/iteration-\(.*\)/\1/p')
        if [ ! -z "$iteration" ]; then
            ITERATIONS+=("$iteration")
        fi
    done

    # Remove duplicates and sort
    ITERATIONS=($(printf "%s\n" "${ITERATIONS[@]}" | sort -u))

    echo "Found ${#ITERATIONS[@]} iterations: ${ITERATIONS[@]}"

    # Submit job for each iteration
    for iteration in "${ITERATIONS[@]}"; do
        job_name="gemini_${run_name}_${inference_prompt_dir}_${EVAL_TARGET}_iter${iteration}"

        echo "Submitting iteration $iteration..."

        sbatch $SLURM_CONFIG \
            --job-name="$job_name" \
            --output="slurm_logging/${job_name}_%j.out" \
            --error="slurm_logging/${job_name}_%j.err" \
            << EOF
#!/bin/bash
#SBATCH --job-name="$job_name"

# Source bash config
source /nas/ucb/nikihowe/config/bashrc
conda activate motivated_reasoning_env

# Change to project directory
cd /nas/ucb/nikihowe/motivated-reasoning

# Run evaluation
python $SCRIPT_PATH \
    --run_name "$run_name" \
    --iteration "$iteration" \
    --inference_prompt_dir "$inference_prompt_dir" \
    --eval_prompt_dir "$EVAL_PROMPT_DIR" \
    --eval_target "$EVAL_TARGET" \
    --evaluator "$EVALUATOR"

echo "Completed evaluation for iteration $iteration"
EOF

    done

    echo "✓ Submitted all ${#ITERATIONS[@]} iterations for $run_name"
done

echo ""
echo "========================================="
echo "ALL JOBS SUBMITTED!"
echo "========================================="
echo ""
echo "Check status with: squeue -u \$USER"
