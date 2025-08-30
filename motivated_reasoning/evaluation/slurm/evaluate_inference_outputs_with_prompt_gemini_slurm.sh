#!/bin/bash

# Script to submit SLURM jobs for evaluating inference outputs using Gemini with customizable evaluation prompts
# Usage: ./evaluate_inference_outputs_with_prompt_gemini_slurm.sh --run_name RUN_NAME --inference_prompt_dir PROMPT --eval_prompt_dir EVAL_PROMPT [other_flags...]
#
# Examples:
#   ./evaluate_inference_outputs_with_prompt_gemini_slurm.sh --run_name harmbench-08_28_213159 --inference_prompt_dir bullet_points_cot --eval_prompt_dir simple_motivated_reasoning
#   ./evaluate_inference_outputs_with_prompt_gemini_slurm.sh --run_name harmbench-08_28_213159 --inference_prompt_dir simple_cot --eval_prompt_dir five_option_first_vs_second
#   ./evaluate_inference_outputs_with_prompt_gemini_slurm.sh --run_name harmbench-08_28_213159 --inference_prompt_dir training_prompt --eval_prompt_dir simple_motivated_reasoning

# Initialize variables
RUN_NAME=""
INFERENCE_PROMPT_DIR=""
EVAL_PROMPT_DIR=""
REMAINING_ARGS=()

# Parse keyword arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --inference_prompt_dir)
            INFERENCE_PROMPT_DIR="$2"
            shift 2
            ;;
        --eval_prompt_dir)
            EVAL_PROMPT_DIR="$2"
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
    echo "Usage: $0 --run_name RUN_NAME --inference_prompt_dir PROMPT --eval_prompt_dir EVAL_PROMPT [other_flags...]"
    exit 1
fi

if [[ -z "$INFERENCE_PROMPT_DIR" ]]; then
    echo "Error: --inference_prompt_dir is required"
    echo "Usage: $0 --run_name RUN_NAME --inference_prompt_dir PROMPT --eval_prompt_dir EVAL_PROMPT [other_flags...]"
    exit 1
fi

if [[ -z "$EVAL_PROMPT_DIR" ]]; then
    echo "Error: --eval_prompt_dir is required"
    echo "Usage: $0 --run_name RUN_NAME --inference_prompt_dir PROMPT --eval_prompt_dir EVAL_PROMPT [other_flags...]"
    exit 1
fi

SCRIPT_PATH="motivated_reasoning/evaluation/local/evaluate_inference_outputs_with_prompt_gemini.py"

# Check if inference directory exists
INFERENCE_PATH="inference_output/$RUN_NAME"
if [ ! -d "$INFERENCE_PATH" ]; then
    echo "Error: Inference directory $INFERENCE_PATH does not exist"
    exit 1
fi

# Check if evaluation prompt directory exists
EVAL_PROMPT_PATH="motivated_reasoning/evaluation/prompts/$EVAL_PROMPT_DIR"
if [ ! -d "$EVAL_PROMPT_PATH" ]; then
    echo "Error: Evaluation prompt directory $EVAL_PROMPT_PATH does not exist"
    echo "Available evaluation prompt directories:"
    ls -1 motivated_reasoning/evaluation/prompts/*/ 2>/dev/null | sed 's/.*\///' | sed 's/\/$//' || echo "  No evaluation prompt directories found"
    exit 1
fi

# Check if prompt.txt and suffix.txt exist
if [ ! -f "$EVAL_PROMPT_PATH/prompt.txt" ]; then
    echo "Error: prompt.txt not found in $EVAL_PROMPT_PATH"
    exit 1
fi

if [ ! -f "$EVAL_PROMPT_PATH/suffix.txt" ]; then
    echo "Error: suffix.txt not found in $EVAL_PROMPT_PATH"
    exit 1
fi

# Find all iteration directories (folders that match iteration-{number} pattern)
ITERATION_DIRS=($(find "$INFERENCE_PATH/$INFERENCE_PROMPT_DIR" -maxdepth 1 -type d -name "iteration-[0-9]*" | sort -V))
if [ ${#ITERATION_DIRS[@]} -eq 0 ]; then
    echo "Error: No iteration directories found in $INFERENCE_PATH/$INFERENCE_PROMPT_DIR"
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
echo "Inference prompt directory: $INFERENCE_PROMPT_DIR"
echo "Evaluation prompt directory: $EVAL_PROMPT_DIR"
if [ ${#REMAINING_ARGS[@]} -gt 0 ]; then
    echo "Additional arguments: ${REMAINING_ARGS[@]}"
fi
echo ""

# SLURM configuration - No GPU needed for Gemini API calls, but need more time for API calls
SLURM_CONFIG="--cpus-per-task=1 --mem=8G --time=0:20:00"

echo "Submitting SLURM jobs for Gemini-based inference output evaluation with custom prompts on iterations: ${ITERATIONS[@]}"
echo "Run name: $RUN_NAME"
echo "Inference prompt directory: $INFERENCE_PROMPT_DIR"
echo "Evaluation prompt directory: $EVAL_PROMPT_DIR"
if [ ${#REMAINING_ARGS[@]} -gt 0 ]; then
    echo "Additional arguments: ${REMAINING_ARGS[@]}"
fi
echo ""

for iteration in "${ITERATIONS[@]}"; do
    echo "Submitting Gemini evaluation job for iteration $iteration..."
    
    # Create job name
    job_name="gemini_eval_${RUN_NAME}_iter${iteration}_${INFERENCE_PROMPT_DIR}_${EVAL_PROMPT_DIR}"
    
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

# Run the Gemini evaluation script with custom prompts
if [ ${#REMAINING_ARGS[@]} -eq 0 ]; then
    python $SCRIPT_PATH \
        --run_name $RUN_NAME \
        --iteration $iteration \
        --inference_prompt_dir $INFERENCE_PROMPT_DIR \
        --eval_prompt_dir $EVAL_PROMPT_DIR
else
    python $SCRIPT_PATH \
        --run_name $RUN_NAME \
        --iteration $iteration \
        --inference_prompt_dir $INFERENCE_PROMPT_DIR \
        --eval_prompt_dir $EVAL_PROMPT_DIR \
        "${REMAINING_ARGS[@]}"
fi

echo "Completed Gemini-based inference output evaluation for iteration $iteration with inference prompt $INFERENCE_PROMPT_DIR and evaluation prompt $EVAL_PROMPT_DIR"
EOF

    echo "Submitted Gemini evaluation job for iteration $iteration with job name: $job_name"
    echo ""
done

echo "All Gemini evaluation SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/"
echo "Evaluation results will be saved in: evaluation_output/$RUN_NAME/$INFERENCE_PROMPT_DIR/evaluator-gemini-25-flash-lite/$EVAL_PROMPT_DIR/iteration-X/"
