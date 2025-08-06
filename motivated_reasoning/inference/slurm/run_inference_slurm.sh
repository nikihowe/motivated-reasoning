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
EXTRA_FLAGS="${@:4}"

MODEL_PATH="/nas/ucb/nikihowe/motivated-reasoning/data/models"
SCRIPT_PATH="motivated_reasoning/inference/run_local/run_inference.py"

# Automatically detect iterations by scanning the model directory
MODEL_DIR="$MODEL_PATH/$RUN_NAME"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR does not exist"
    exit 1
fi

# Find all iteration directories (folders that are numbers)
ITERATIONS=($(find "$MODEL_DIR" -maxdepth 1 -type d -name "[0-9]*" | sort -n | xargs -n1 basename))
if [ ${#ITERATIONS[@]} -eq 0 ]; then
    echo "Error: No iteration directories found in $MODEL_DIR"
    exit 1
fi

echo "Found ${#ITERATIONS[@]} iterations: ${ITERATIONS[@]}"

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
    $EXTRA_FLAGS

echo "Completed inference for iteration $iteration"
EOF

    echo "Submitted job for iteration $iteration with job name: $job_name"
    echo ""
done

echo "All SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/" 