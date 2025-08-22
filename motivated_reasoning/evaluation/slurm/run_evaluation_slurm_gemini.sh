#!/bin/bash

# Script to submit SLURM jobs for evaluating multiple iterations of motivated reasoning using Gemini
# Usage: ./run_evaluation_slurm_gemini.sh --inference_dir DIR [--prompt_type TYPE]
#
# Examples:
#   ./run_evaluation_slurm_gemini.sh --inference_dir my_experiment                    # Use cot_prompt (default)
#   ./run_evaluation_slurm_gemini.sh --inference_dir my_experiment --prompt_type training_prompt
#   ./run_evaluation_slurm_gemini.sh --inference_dir my_experiment --prompt_type cot_prompt

# Default values
DEFAULT_INFERENCE_DIR="harmbench_cot_tags-08_18_162853"
DEFAULT_PROMPT_TYPE="cot_prompt"

# Initialize variables
INFERENCE_DIR=""
PROMPT_TYPE="$DEFAULT_PROMPT_TYPE"

# Parse keyword arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --inference_dir)
            INFERENCE_DIR="$2"
            shift 2
            ;;
        --prompt_type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --inference_dir DIR [--prompt_type TYPE]"
            exit 1
            ;;
    esac
done

# Use default inference directory if not provided
INFERENCE_DIR="${INFERENCE_DIR:-$DEFAULT_INFERENCE_DIR}"

# Validate required arguments
if [[ -z "$INFERENCE_DIR" ]]; then
    echo "Error: --inference_dir is required"
    echo "Usage: $0 --inference_dir DIR [--prompt_type TYPE]"
    exit 1
fi

SCRIPT_PATH="motivated_reasoning/evaluation/local/evaluate_motivated_cots_gemini.py"

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
echo "Prompt type: $PROMPT_TYPE"
echo ""

# SLURM configuration - No GPU needed for Gemini API calls, but need more time for API calls
SLURM_CONFIG="--cpus-per-task=2 --mem=8G --time=0:20:00"

echo "Submitting SLURM jobs for Gemini-based motivated reasoning evaluation on iterations: ${ITERATIONS[@]}"
echo "Inference directory: $INFERENCE_DIR"
echo "Prompt type: $PROMPT_TYPE"
echo ""

for iteration in "${ITERATIONS[@]}"; do
    echo "Submitting Gemini evaluation job for iteration $iteration..."
    
    # Create job name
    job_name="gemini_eval_${INFERENCE_DIR}_iter${iteration}"
    
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

# Run the Gemini evaluation script - no evaluator_iteration needed since we use Gemini directly
python $SCRIPT_PATH --directory $INFERENCE_DIR --iteration $iteration --prompt_type $PROMPT_TYPE

echo "Completed Gemini-based motivated reasoning evaluation for iteration $iteration"
EOF

    echo "Submitted Gemini evaluation job for iteration $iteration with job name: $job_name"
    echo ""
done

echo "All Gemini evaluation SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/"
echo "Evaluation results will be saved in: evaluation_output/$INFERENCE_DIR/evaluator-gemini/iteration-X/$PROMPT_TYPE/"
