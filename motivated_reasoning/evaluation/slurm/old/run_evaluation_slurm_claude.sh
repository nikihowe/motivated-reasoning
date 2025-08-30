#!/bin/bash

# Script to submit SLURM jobs for evaluating multiple iterations of motivated reasoning using Claude
# Usage: ./run_evaluation_slurm_claude.sh [inference_dir]
#
# Examples:
#   ./run_evaluation_slurm_claude.sh                                  # Use default inference directory
#   ./run_evaluation_slurm_claude.sh my_experiment                    # Evaluate my_experiment directory
#   ./run_evaluation_slurm_claude.sh harmbench_cot_tags-08_18_162853  # Evaluate specific experiment

# Default inference directory
DEFAULT_INFERENCE_DIR="harmbench_cot_tags-08_18_162853"

# Use provided inference directory or default
INFERENCE_DIR="${1:-$DEFAULT_INFERENCE_DIR}"

SCRIPT_PATH="motivated_reasoning/evaluation/local/evaluate_motivated_cots_claude.py"

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
echo ""

# SLURM configuration - No GPU needed for Claude API calls, but need more time for API calls
SLURM_CONFIG="--cpus-per-task=2 --mem=8G --time=0:20:00"

echo "Submitting SLURM jobs for Claude-based motivated reasoning evaluation on iterations: ${ITERATIONS[@]}"
echo "Inference directory: $INFERENCE_DIR"
echo ""

for iteration in "${ITERATIONS[@]}"; do
    echo "Submitting Claude evaluation job for iteration $iteration..."
    
    # Create job name
    job_name="claude_eval_${INFERENCE_DIR}_iter${iteration}"
    
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

# Run the Claude evaluation script - no evaluator_iteration needed since we use Claude directly
python $SCRIPT_PATH --directory $INFERENCE_DIR --iteration $iteration

echo "Completed Claude-based motivated reasoning evaluation for iteration $iteration"
EOF

    echo "Submitted Claude evaluation job for iteration $iteration with job name: $job_name"
    echo ""
done

echo "All Claude evaluation SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/"
echo "Evaluation results will be saved in: evaluation_output/$INFERENCE_DIR/evaluator-claude/"
