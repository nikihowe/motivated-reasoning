#!/bin/bash

# Script to submit SLURM jobs for multiple iterations of inference
# Usage: ./run_inference_slurm.sh --run_name RUN_NAME [--dataset_type {train,test}] [other_flags...]
# Examples:
#   ./run_inference_slurm.sh --run_name my_model
#   ./run_inference_slurm.sh --run_name my_model --dataset_type train
#   ./run_inference_slurm.sh --run_name my_model --use_training_prompt
#   ./run_inference_slurm.sh --run_name my_model --dataset_type test --use_training_prompt --add_true_reasoning_suffix

# Default values
DEFAULT_RUN_NAME="harmbench_kto_long_lr_5e-5-06_20_113158"
DEFAULT_DATASET_TYPE="test"

# Initialize variables
RUN_NAME=""
DATASET_TYPE="$DEFAULT_DATASET_TYPE"

# Parse all arguments as flags
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --dataset_type)
            DATASET_TYPE="$2"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Use default run_name if not provided
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"

# Validate required arguments
if [[ -z "$RUN_NAME" ]]; then
    echo "Error: --run_name is required"
    echo "Usage: $0 --run_name RUN_NAME [--dataset_type {train,test}] [other_flags...]"
    exit 1
fi

MODEL_PATH="/nas/ucb/nikihowe/motivated-reasoning/data/models"
SCRIPT_PATH="motivated_reasoning/inference/run_local/run_inference.py"

# Determine suffix string based on flags; default to no_suffix
ADD_TRUE_REASONING=0
ADD_NON_HARMFUL=0
USE_TRAINING_PROMPT=0
ONLY_MISSING=1  # default behavior: only run iterations with no outputs for this suffix

# Filter out script-only flags like --force (not passed to Python)
FILTERED_ARGS=()
for arg in "${REMAINING_ARGS[@]}"; do
    case "$arg" in
        --add_true_reasoning_suffix)
            ADD_TRUE_REASONING=1
            FILTERED_ARGS+=("$arg")
            ;;
        --add_non_harmful_suffix)
            ADD_NON_HARMFUL=1
            FILTERED_ARGS+=("$arg")
            ;;
        --use_training_prompt)
            USE_TRAINING_PROMPT=1
            FILTERED_ARGS+=("$arg")
            ;;
        --force)
            ONLY_MISSING=0
            # do not forward to python
            ;;
        *)
            FILTERED_ARGS+=("$arg")
            ;;
    esac
done

# Determine prompt type directory
if [ $USE_TRAINING_PROMPT -eq 1 ]; then
    PROMPT_TYPE_DIR="training_prompt"
else
    PROMPT_TYPE_DIR="cot_prompt"
fi

SUFFIX_PARTS=()
if [ $ADD_TRUE_REASONING -eq 1 ]; then SUFFIX_PARTS+=("true_reasoning"); fi
if [ $ADD_NON_HARMFUL -eq 1 ]; then SUFFIX_PARTS+=("non_harmful"); fi
if [ ${#SUFFIX_PARTS[@]} -eq 0 ]; then
    SUFFIX_STR="no_suffix"
else
    SUFFIX_STR="$(IFS=_; echo "${SUFFIX_PARTS[*]}")"
fi

# Keep FILTERED_ARGS as an array for proper argument passing

# Automatically detect iterations by scanning the model directory
MODEL_DIR="$MODEL_PATH/$RUN_NAME"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR does not exist"
    exit 1
fi

# Find all iteration directories (folders that are numbers) and sort numerically
AVAILABLE_ITERATIONS=($(find "$MODEL_DIR" -maxdepth 1 -type d -name "[0-9]*" | xargs -n1 basename | sort -n))
if [ ${#AVAILABLE_ITERATIONS[@]} -eq 0 ]; then
    echo "Error: No iteration directories found in $MODEL_DIR"
    exit 1
fi

echo "Found ${#AVAILABLE_ITERATIONS[@]} iterations: ${AVAILABLE_ITERATIONS[@]}"

# Environment name is auto-detected by the Python script, no need to duplicate logic here

# Decide which iterations to run based on existing outputs
if [ $ONLY_MISSING -eq 1 ]; then
    echo "Selecting only iterations missing outputs for prompt type '$PROMPT_TYPE_DIR' and suffix '$SUFFIX_STR'"
    ITERATIONS=()
    for it in "${AVAILABLE_ITERATIONS[@]}"; do
        OUT_DIR="inference_output/$RUN_NAME/iteration-$it/$PROMPT_TYPE_DIR/$SUFFIX_STR"
        if compgen -G "$OUT_DIR/*.jsonl" > /dev/null; then
            echo "Skipping iteration $it (outputs already exist in $OUT_DIR)"
        else
            ITERATIONS+=("$it")
        fi
    done
else
    echo "--force specified: running all iterations"
    ITERATIONS=("${AVAILABLE_ITERATIONS[@]}")
fi

if [ ${#ITERATIONS[@]} -eq 0 ]; then
    echo "No iterations to run. Exiting."
    exit 0
fi

# SLURM configuration
# SLURM_CONFIG="--partition=main --gpus=A6000:1 --cpus-per-task=4 --mem=32G --time=0:20:00"
SLURM_CONFIG="--partition=main --gpus=1 --cpus-per-task=4 --mem=32G --time=0:20:00"

mkdir -p slurm_logging

echo "Submitting SLURM jobs for iterations: ${ITERATIONS[@]}"
echo "Model: $RUN_NAME"
echo "Dataset type: $DATASET_TYPE"
echo "Model path: $MODEL_PATH"
echo "Prompt type: $PROMPT_TYPE_DIR"
echo "Suffix: $SUFFIX_STR"

echo ""

for iteration in "${ITERATIONS[@]}"; do
    echo "Submitting job for iteration $iteration..."

    # Create descriptive job name with prompt type
    prompt_suffix=$([ $USE_TRAINING_PROMPT -eq 1 ] && echo "_training" || echo "_cot")
    job_name="infer_${RUN_NAME}_iter${iteration}${prompt_suffix}"

    sbatch $SLURM_CONFIG \
        --job-name="$job_name" \
        --output="slurm_logging/${job_name}_%j.out" \
        --error="slurm_logging/${job_name}_%j.err" \
        << EOF
#!/bin/bash
#SBATCH --job-name="$job_name"

# Source bash config and conda
source /nas/ucb/nikihowe/config/bashrc
source /nas/ucb/nikihowe/miniconda3/etc/profile.d/conda.sh
conda activate motivated_reasoning_env

# Change to project directory
cd /nas/ucb/nikihowe/motivated-reasoning

# Run the inference script (env_name is now auto-detected)
python $SCRIPT_PATH \
    --run_name $RUN_NAME \
    --iteration $iteration \
    --dataset_type $DATASET_TYPE \
    --model_path $MODEL_PATH \
    "${FILTERED_ARGS[@]}"

echo "Completed inference for iteration $iteration"
EOF

    echo "Submitted job for iteration $iteration with job name: $job_name"
    echo ""
done

echo "All SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/"