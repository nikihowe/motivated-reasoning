#!/bin/bash

# Script to submit SLURM jobs for evaluating inference outputs using Gemini
# Usage: ./evaluate_inference_outputs_gemini_slurm.sh --run_name RUN_NAME --inference_prompt PROMPT [other_flags...]
#
# Examples:
#   ./evaluate_inference_outputs_gemini_slurm.sh --run_name harmbench-08_28_213159 --inference_prompt bullet_points_cot
#   ./evaluate_inference_outputs_gemini_slurm.sh --run_name harmbench-08_28_213159 --inference_prompt simple_cot
#   ./evaluate_inference_outputs_gemini_slurm.sh --run_name harmbench-08_28_213159 --inference_prompt training_prompt

# Initialize variables
RUN_NAME=""
INFERENCE_PROMPT=""
REMAINING_ARGS=()

# Parse keyword arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --inference_prompt)
            INFERENCE_PROMPT="$2"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$RUN_NAME" ]]; then
    echo "Error: --run_name is required"
    echo "Usage: $0 --run_name RUN_NAME --inference_prompt PROMPT [other_flags...]"
    exit 1
fi

if [[ -z "$INFERENCE_PROMPT" ]]; then
    echo "Error: --inference_prompt is required"
    echo "Usage: $0 --run_name RUN_NAME --inference_prompt PROMPT [other_flags...]"
    exit 1
fi

SCRIPT_PATH="motivated_reasoning/evaluation/local/evaluate_inference_outputs_gemini.py"

# Check if inference directory exists
INFERENCE_PATH="inference_output/$RUN_NAME"
if [ ! -d "$INFERENCE_PATH" ]; then
    echo "Error: Inference directory $INFERENCE_PATH does not exist"
    exit 1
fi

# Find all iteration directories (folders that match iteration-{number} pattern)
ITERATION_DIRS=($(find "$INFERENCE_PATH" -maxdepth 1 -type d -name "iteration-[0-9]*" | sort -V))
if [ ${#ITERATION_DIRS[@]} -eq 0 ]; then
    echo "Error: No iteration directories found in $INFERENCE_PATH"
    exit 1
fi

# Extract iteration numbers from directory names
ITERATIONS=()
for dir in "${ITERATION_DIRS[@]}"; do
    dirname=$(basename "$dir")
    # Extract iteration number from directory name like "iteration-7"
    iteration=$(echo "$dirname" | sed -n 's/iteration-\([0-9]*\)/\1/p')
    if [ ! -z "$iteration" ]; then
        ITERATIONS+=("$iteration")
    fi
done

# Remove duplicates and sort
ITERATIONS=($(printf "%s\n" "${ITERATIONS[@]}" | sort -nu))

if [ ${#ITERATIONS[@]} -eq 0 ]; then
    echo "Error: No valid iteration numbers found in $INFERENCE_PATH"
    exit 1
fi

echo "Found ${#ITERATIONS[@]} iterations: ${ITERATIONS[@]}"
echo "Run name: $RUN_NAME"
echo "Inference prompt: $INFERENCE_PROMPT"
if [ ${#REMAINING_ARGS[@]} -gt 0 ]; then
    echo "Additional arguments: ${REMAINING_ARGS[@]}"
fi
echo ""

# SLURM configuration - No GPU needed for Gemini API calls, but need more time for API calls
SLURM_CONFIG="--cpus-per-task=1 --mem=8G --time=0:20:00"

echo "Submitting SLURM jobs for Gemini-based inference output evaluation on iterations: ${ITERATIONS[@]}"
echo "Run name: $RUN_NAME"
echo "Inference prompt: $INFERENCE_PROMPT"
if [ ${#REMAINING_ARGS[@]} -gt 0 ]; then
    echo "Additional arguments: ${REMAINING_ARGS[@]}"
fi
echo ""

for iteration in "${ITERATIONS[@]}"; do
    echo "Submitting Gemini evaluation job for iteration $iteration..."
    
    # Create job name
    job_name="gemini_eval_${RUN_NAME}_iter${iteration}_${INFERENCE_PROMPT}"
    
    # Submit SLURM job
    sbatch $SLURM_CONFIG \
        --job-name="$job_name" \
        --output="slurm_logging/${job_name}_%j.out" \
        --error="slurm_logging/${job_name}_%j.err" \
        << EOF
#!/bin/bash
#SBATCH --job-name="$job_name"

# Source bash config, which also does conda
source /nas/ucb/nikihowe/config/bashrc
conda activate motivated_reasoning_env

# Change to project directory
cd /nas/ucb/nikihowe/motivated-reasoning

# Run the Gemini evaluation script
if [ ${#REMAINING_ARGS[@]} -eq 0 ]; then
    python $SCRIPT_PATH \
        --run_name $RUN_NAME \
        --iteration $iteration \
        --inference_prompt $INFERENCE_PROMPT
else
    python $SCRIPT_PATH \
        --run_name $RUN_NAME \
        --iteration $iteration \
        --inference_prompt $INFERENCE_PROMPT \
        "${REMAINING_ARGS[@]}"
fi

echo "Completed Gemini-based inference output evaluation for iteration $iteration with prompt $INFERENCE_PROMPT"
EOF

    echo "Submitted Gemini evaluation job for iteration $iteration with job name: $job_name"
    echo ""
done

echo "All Gemini evaluation SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/"
echo "Evaluation results will be saved in: evaluation_output/$RUN_NAME/evaluator-gemini-25-flash-lite/iteration-X/$INFERENCE_PROMPT/"
