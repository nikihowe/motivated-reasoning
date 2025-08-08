#!/bin/bash

# Script to submit SLURM jobs for multiple iterations of inference
# Usage: ./run_multiple_iterations_slurm.sh [run_name] [env_name] [dataset_type] [extra_flags...]

# Default values
DEFAULT_RUN_NAME="harmbench_kto_long_lr_5e-5-06_20_113158"
DEFAULT_ENV_NAME="harmbench"
DEFAULT_DATASET_TYPE="test"

# Parse arguments
RUN_NAME="${1:-$DEFAULT_RUN_NAME}"
ENV_NAME="${2:-$DEFAULT_ENV_NAME}"
DATASET_TYPE="${3:-$DEFAULT_DATASET_TYPE}"

# Accept extra flags for python script (everything after the first 3 arguments)
# Default: run only iterations without outputs for the chosen suffix
# Use --force to run all iterations regardless of existing outputs
REMAINING_ARGS=("${@:4}")
ONLY_MISSING=1
TRUE_REASONING_FLAG=0
NON_HARMFUL_FLAG=0
EXTRA_FLAGS=()
for arg in "${REMAINING_ARGS[@]}"; do
    case "$arg" in
        --force) ONLY_MISSING=0 ;;
        --add_true_reasoning_suffix) TRUE_REASONING_FLAG=1; EXTRA_FLAGS+=("$arg") ;;
        --add_non_harmful_suffix) NON_HARMFUL_FLAG=1; EXTRA_FLAGS+=("$arg") ;;
        *) EXTRA_FLAGS+=("$arg") ;;
    esac
done
# Determine suffix string to check existing outputs
SUFFIX_PARTS=()
if [ $TRUE_REASONING_FLAG -eq 1 ]; then SUFFIX_PARTS+=("true_reasoning"); fi
if [ $NON_HARMFUL_FLAG -eq 1 ]; then SUFFIX_PARTS+=("non_harmful"); fi
if [ ${#SUFFIX_PARTS[@]} -eq 0 ]; then
    SUFFIX_STR="no_suffix"
else
    IFS='_'; SUFFIX_STR="${SUFFIX_PARTS[*]}"; unset IFS
fi
# Join remaining extra flags into a single string for passing through
EXTRA_FLAGS_STR="${EXTRA_FLAGS[*]}"

MODEL_PATH="/nas/ucb/nikihowe/motivated-reasoning/data/models"
SCRIPT_PATH="motivated_reasoning/inference/run_local/run_inference.py"

# Automatically detect iterations by scanning the model directory
MODEL_DIR="$MODEL_PATH/$RUN_NAME"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR does not exist"
    exit 1
fi

# Find available iteration directories (folders that are numbers)
AVAILABLE_ITERATIONS=($(find "$MODEL_DIR" -maxdepth 1 -type d -name "[0-9]*" | sort -n | xargs -n1 basename))
if [ ${#AVAILABLE_ITERATIONS[@]} -eq 0 ]; then
    echo "Error: No iteration directories found in $MODEL_DIR"
    exit 1
fi

echo "Available iterations: ${AVAILABLE_ITERATIONS[@]}"

# Start from all available iterations
CANDIDATE_ITERATIONS=("${AVAILABLE_ITERATIONS[@]}")

# If ONLY_MISSING, remove iterations that already have output files for this suffix
if [ $ONLY_MISSING -eq 1 ]; then
    ITERATIONS=()
    for it in "${CANDIDATE_ITERATIONS[@]}"; do
        OUT_DIR="inference_output/$RUN_NAME/iteration-$it/$SUFFIX_STR"
        if compgen -G "$OUT_DIR/*.jsonl" > /dev/null; then
            echo "Skipping iteration $it (outputs already exist in $OUT_DIR)"
        else
            ITERATIONS+=("$it")
        fi
    done
else
    ITERATIONS=("${CANDIDATE_ITERATIONS[@]}")
fi

if [ ${#ITERATIONS[@]} -eq 0 ]; then
    echo "Nothing to run. Exiting."
    exit 0
fi

echo "Running for iterations: ${ITERATIONS[@]}"

# SLURM configuration
SLURM_CONFIG="--partition=main --gpus=A6000:1 --cpus-per-task=4 --mem=32G --time=0:20:00"

echo "Submitting SLURM jobs for iterations: ${ITERATIONS[@]}"
echo "Model: $RUN_NAME"
echo "Environment: $ENV_NAME"
echo "Dataset type: $DATASET_TYPE"
echo "Model path: $MODEL_PATH"
echo ""

for iteration in "${ITERATIONS[@]}"; do
    echo "Submitting job for iteration $iteration..."
    
    # Create job name
    job_name="infer_${RUN_NAME}_iter${iteration}"
    
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

# Run the inference script
python $SCRIPT_PATH \
    --run_name $RUN_NAME \
    --iteration $iteration \
    --env_name $ENV_NAME \
    --dataset_type $DATASET_TYPE \
    --model_path $MODEL_PATH \
    $EXTRA_FLAGS_STR

echo "Completed inference for iteration $iteration"
EOF

    echo "Submitted job for iteration $iteration with job name: $job_name"
    echo ""
done

echo "All SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/" 