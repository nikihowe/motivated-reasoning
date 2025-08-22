#!/bin/bash

# Script to submit SLURM jobs for evaluating multiple iterations of HarmBench inference
# Usage: ./run_evaluation_slurm.sh --inference_dir DIR [other_flags...]
#
# Examples:
#   ./run_evaluation_slurm.sh --inference_dir my_experiment
#   ./run_evaluation_slurm.sh --inference_dir my_experiment --evaluator_iteration 8
#   ./run_evaluation_slurm.sh --inference_dir my_experiment --prompt_type training_prompt
#   ./run_evaluation_slurm.sh --inference_dir my_experiment --evaluator_iteration 8 --prompt_type training_prompt

# Initialize variables
INFERENCE_DIR=""
REMAINING_ARGS=()

# Parse keyword arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --inference_dir)
            INFERENCE_DIR="$2"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# No fallback to default - inference_dir is always required

# Validate required arguments
if [[ -z "$INFERENCE_DIR" ]]; then
    echo "Error: --inference_dir is required"
    echo "Usage: $0 --inference_dir DIR [other_flags...]"
    exit 1
fi

SCRIPT_PATH="motivated_reasoning/evaluation/local/evaluate_motivated_cots.py"

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
if [ ${#REMAINING_ARGS[@]} -gt 0 ]; then
    echo "Additional arguments: ${REMAINING_ARGS[@]}"
fi
echo ""

# SLURM configuration
# SLURM_CONFIG="--partition=main --gpus=A6000:1 --cpus-per-task=4 --mem=32G --time=0:30:00"
SLURM_CONFIG="--gpus=1 --mem=16G --time=0:10:00"

echo "Submitting SLURM jobs for evaluating iterations: ${ITERATIONS[@]}"
echo "Inference directory: $INFERENCE_DIR"
if [ ${#REMAINING_ARGS[@]} -gt 0 ]; then
    echo "Additional arguments: ${REMAINING_ARGS[@]}"
fi
echo ""

for iteration in "${ITERATIONS[@]}"; do
    echo "Submitting evaluation job for iteration $iteration..."
    
    # Create job name
    job_name="eval_${INFERENCE_DIR}_iter${iteration}"
    
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

# Run the evaluation script
if [ ${#REMAINING_ARGS[@]} -eq 0 ]; then
    python $SCRIPT_PATH \
        --directory $INFERENCE_DIR \
        --iteration $iteration
else
    python $SCRIPT_PATH \
        --directory $INFERENCE_DIR \
        --iteration $iteration \
        "${REMAINING_ARGS[@]}"
fi

echo "Completed evaluation for iteration $iteration"
EOF

    echo "Submitted evaluation job for iteration $iteration with job name: $job_name"
    echo ""
done

echo "All evaluation SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/"
echo "Evaluation results will be saved in: evaluation_output/$INFERENCE_DIR/evaluator-$EVALUATOR_ITERATION/iteration-X/$PROMPT_TYPE/" 