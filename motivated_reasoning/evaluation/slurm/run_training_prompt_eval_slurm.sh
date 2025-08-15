#!/bin/bash

# Script to submit SLURM jobs for evaluating multiple iterations of HarmBench inference for unreasonable justifications
# Usage: ./run_training_prompt_eval_slurm.sh --inference_dir DIR [--evaluator_iteration ITER] [--prompt_type TYPE]
#
# Examples:
#   ./run_training_prompt_eval_slurm.sh --inference_dir my_experiment
#   ./run_training_prompt_eval_slurm.sh --inference_dir my_experiment --evaluator_iteration 8
#   ./run_training_prompt_eval_slurm.sh --inference_dir my_experiment --prompt_type cot_prompt
#   ./run_training_prompt_eval_slurm.sh --inference_dir my_experiment --evaluator_iteration base --prompt_type training_prompt

# Default values
DEFAULT_INFERENCE_DIR="harmbench_kto_long_lr_5e-5-06_20_113158"
DEFAULT_EVALUATOR_ITERATION="base"
DEFAULT_PROMPT_TYPE="training_prompt"

# Initialize variables
INFERENCE_DIR=""
EVALUATOR_ITERATION="$DEFAULT_EVALUATOR_ITERATION"
PROMPT_TYPE="$DEFAULT_PROMPT_TYPE"

# Parse keyword arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --inference_dir)
            INFERENCE_DIR="$2"
            shift 2
            ;;
        --evaluator_iteration)
            EVALUATOR_ITERATION="$2"
            shift 2
            ;;
        --prompt_type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --inference_dir DIR [--evaluator_iteration ITER] [--prompt_type TYPE]"
            exit 1
            ;;
    esac
done

# Use default inference directory if not provided
INFERENCE_DIR="${INFERENCE_DIR:-$DEFAULT_INFERENCE_DIR}"

# Validate required arguments
if [[ -z "$INFERENCE_DIR" ]]; then
    echo "Error: --inference_dir is required"
    echo "Usage: $0 --inference_dir DIR [--evaluator_iteration ITER] [--prompt_type TYPE]"
    exit 1
fi

SCRIPT_PATH="motivated_reasoning/evaluation/local/evaluate_training_prompt_responses.py"

# Check if inference directory exists
INFERENCE_PATH="inference_output/$INFERENCE_DIR"
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
echo "Inference directory: $INFERENCE_DIR"
echo "Evaluator iteration: $EVALUATOR_ITERATION"
echo "Prompt type: $PROMPT_TYPE"
echo ""

# SLURM configuration
SLURM_CONFIG="--partition=main --gpus=A6000:1 --cpus-per-task=4 --mem=32G --time=0:15:00"

echo "Submitting SLURM jobs for evaluating training prompt responses on iterations: ${ITERATIONS[@]}"
echo "Inference directory: $INFERENCE_DIR"
echo "Evaluator iteration: $EVALUATOR_ITERATION"
echo "Prompt type: $PROMPT_TYPE"
echo ""

for iteration in "${ITERATIONS[@]}"; do
    echo "Submitting training prompt evaluation job for iteration $iteration..."
    
    # Create job name
    if [ "$EVALUATOR_ITERATION" = "base" ]; then
        job_name="training_prompt_eval_${INFERENCE_DIR}_iter${iteration}_base"
    else
        job_name="training_prompt_eval_${INFERENCE_DIR}_iter${iteration}_eval${EVALUATOR_ITERATION}"
    fi
    
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

# Run the evaluation script with new argument format
if [ "$EVALUATOR_ITERATION" = "base" ]; then
    python $SCRIPT_PATH --directory $INFERENCE_DIR --iteration $iteration --prompt_type $PROMPT_TYPE
else
    python $SCRIPT_PATH --directory $INFERENCE_DIR --iteration $iteration --evaluator_iteration $EVALUATOR_ITERATION --prompt_type $PROMPT_TYPE
fi

echo "Completed training prompt evaluation for iteration $iteration"
EOF

    echo "Submitted training prompt evaluation job for iteration $iteration with job name: $job_name"
    echo ""
done

echo "All training prompt evaluation SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/"
echo "Evaluation results will be saved in: evaluation_output/recommendation_classification/$INFERENCE_DIR/"
