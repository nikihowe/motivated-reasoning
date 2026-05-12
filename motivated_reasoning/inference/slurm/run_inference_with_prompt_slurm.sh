#!/bin/bash

# Script to submit SLURM jobs for base model + multiple iterations of inference with custom prompts
# Usage: ./run_inference_with_prompt_slurm.sh --run_name RUN_NAME --prompt_file PROMPT_FILE [--dataset_type {train,test}] [other_flags...]
# 
# By default, only runs jobs for missing outputs (base model + iterations with no existing outputs)
# Use --force to re-run all jobs regardless of existing outputs
#
# Examples:
#   ./run_inference_with_prompt_slurm.sh --run_name my_model --prompt_file bullet_points_cot
#   ./run_inference_with_prompt_slurm.sh --run_name my_model --prompt_file simple_cot --dataset_type train
#   ./run_inference_with_prompt_slurm.sh --run_name my_model --prompt_file bullet_points_cot --test
#   ./run_inference_with_prompt_slurm.sh --run_name my_model --prompt_file simple_cot --force

RUN_NAME=""
PROMPT_FILE=""
SLURM_TIME="0:10:00"
SLURM_MEM="24G"
REMAINING_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --prompt_file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        --slurm_time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --slurm_mem)
            SLURM_MEM="$2"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Use default values if not provided
RUN_NAME="${RUN_NAME:-}"
PROMPT_FILE="${PROMPT_FILE:-}"

# Validate required arguments
if [[ -z "$RUN_NAME" ]]; then
    echo "Error: --run_name is required"
    echo "Usage: $0 --run_name RUN_NAME --prompt_file PROMPT_FILE [other_flags...]"
    exit 1
fi

if [[ -z "$PROMPT_FILE" ]]; then
    echo "Error: --prompt_file is required"
    echo "Usage: $0 --run_name RUN_NAME --prompt_file PROMPT_FILE [other_flags...]"
    exit 1
fi

SCRIPT_PATH="motivated_reasoning/inference/local/run_inference_with_prompt.py"

# Check if prompt file exists
PROMPT_FILE_PATH="motivated_reasoning/inference/prompts/${PROMPT_FILE}.txt"
if [ ! -f "$PROMPT_FILE_PATH" ]; then
    echo "Error: Prompt file $PROMPT_FILE_PATH does not exist"
    echo "Available prompt files:"
    ls -1 motivated_reasoning/inference/prompts/*.txt | sed 's/.*\///' | sed 's/\.txt$//'
    exit 1
fi

ONLY_MISSING=1  # default behavior: only run iterations with no outputs for this prompt

# Filter out script-only flags like --force (not passed to Python)
FILTERED_ARGS=()
for arg in "${REMAINING_ARGS[@]}"; do
    case "$arg" in
        --force)
            ONLY_MISSING=0
            # do not forward to python
            ;;
        *)
            FILTERED_ARGS+=("$arg")
            ;;
    esac
done

# Keep FILTERED_ARGS as an array for proper argument passing

# Automatically detect iterations by scanning the model directory
MODEL_DIR="/nas/ucb/nikihowe/motivated-reasoning/data/models/$RUN_NAME"
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

# Decide which iterations to run based on existing outputs
# Create unified list of all iterations (base + numeric)
ALL_ITERATIONS=("base" "${AVAILABLE_ITERATIONS[@]}")

if [ $ONLY_MISSING -eq 1 ]; then
    echo "Selecting only iterations missing outputs for prompt '$PROMPT_FILE'"
    ITERATIONS=()
    
    # Check all iterations uniformly
    for it in "${ALL_ITERATIONS[@]}"; do
        OUT_DIR="inference_output/$RUN_NAME/$PROMPT_FILE/iteration-$it"
        if [ -d "$OUT_DIR" ] && compgen -G "$OUT_DIR/*.jsonl" > /dev/null; then
            echo "Skipping iteration $it (outputs already exist in $OUT_DIR)"
        else
            ITERATIONS+=("$it")
        fi
    done
else
    echo "--force specified: running all iterations"
    ITERATIONS=("${ALL_ITERATIONS[@]}")
fi

if [ ${#ITERATIONS[@]} -eq 0 ]; then
    echo "No jobs to run (all outputs already exist). Exiting."
    exit 0
fi

# Get GPU node configuration from GPU groups file
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJ_DIR="$( dirname "$( dirname "$( dirname "$SCRIPT_DIR" )" )" )"
NODE_LIST=$(grep "^noshards=" "$PROJ_DIR/gpu_groups.txt" | cut -d'=' -f2)

if [ -z "$NODE_LIST" ]; then
    echo "Error: Could not find noshards group in gpu_groups.txt"
    exit 1
fi

mkdir -p slurm_logging

echo "Submitting SLURM jobs for: ${ITERATIONS[@]}"
echo "Model: $RUN_NAME"
echo "Prompt file: $PROMPT_FILE"
echo "SLURM time: $SLURM_TIME"
echo "SLURM mem: $SLURM_MEM"

echo ""

for iteration in "${ITERATIONS[@]}"; do
    echo "Submitting job for iteration $iteration..."
    job_name="infer_${RUN_NAME}_iter${iteration}_${PROMPT_FILE}"
    model_args="--iteration $iteration"
    completion_msg="Completed inference for iteration $iteration with prompt $PROMPT_FILE"
    extra_args_string=""
    if [ ${#FILTERED_ARGS[@]} -gt 0 ]; then
        extra_args_string="$(printf ' %q' "${FILTERED_ARGS[@]}")"
    fi
    
    # Create batch script file
    batch_script="slurm_logging/${job_name}.sh"
    
    cat << EOF > "$batch_script"
#!/bin/bash
#SBATCH --job-name="$job_name"
#SBATCH --output="slurm_logging/${job_name}_%j.out"
#SBATCH --error="slurm_logging/${job_name}_%j.err"
#SBATCH --gpus=1
#SBATCH --mem=$SLURM_MEM
#SBATCH --time=$SLURM_TIME
#SBATCH --nodes=1
#SBATCH --nodelist=$NODE_LIST

set -euo pipefail

# Source bash config and conda
source /nas/ucb/nikihowe/config/bashrc
source /nas/ucb/nikihowe/miniconda3/etc/profile.d/conda.sh
conda activate motivated_reasoning_env

# Change to project directory
cd /nas/ucb/nikihowe/motivated-reasoning

# Run the inference script with custom prompt
python $SCRIPT_PATH \
    --run_name $RUN_NAME \
    $model_args \
    --prompt_file $PROMPT_FILE$extra_args_string

echo "$completion_msg"
EOF

    # Submit the batch script
    sbatch "$batch_script"

    echo "Submitted job for $iteration with job name: $job_name"
    echo ""
done

echo "All SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/"
