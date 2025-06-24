#!/bin/bash

# Script to submit SLURM jobs for multiple iterations of HarmBench inference
# Usage: ./run_multiple_iterations_slurm.sh

RUN_NAME="harmbench_kto_long_lr_5e-5-06_20_113158"
MODEL_PATH="/nas/ucb/nikihowe/chai_motivated_reasoning/data/models"
SCRIPT_PATH="targeted_llm_manipulation/local_inference/run_harmbench_inference.py"

# Define iterations to run
ITERATIONS=({0..5})

# SLURM configuration
SLURM_CONFIG="--partition=main --gres=gpu:2 --cpus-per-task=4 --mem=32G --time=0:30:00"

echo "Submitting SLURM jobs for iterations: ${ITERATIONS[@]}"
echo "Model: $RUN_NAME"
echo "Model path: $MODEL_PATH"
echo ""

for iteration in "${ITERATIONS[@]}"; do
    echo "Submitting job for iteration $iteration..."
    
    # Create job name
    job_name="harmbench_${RUN_NAME}_iter${iteration}"
    
    # Submit SLURM job
    sbatch $SLURM_CONFIG \
        --job-name="$job_name" \
        --output="slurm_logging/${job_name}_%j.out" \
        --error="slurm_logging/${job_name}_%j.err" \
        << EOF
#!/bin/bash
#SBATCH --job-name="$job_name"

# Activate conda environment if needed
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env_name

# Change to project directory
cd /nas/ucb/nikihowe/chai_motivated_reasoning

# Run the inference script
python $SCRIPT_PATH \
    --run_name $RUN_NAME \
    --iteration $iteration \
    --model_path $MODEL_PATH

echo "Completed inference for iteration $iteration"
EOF

    echo "Submitted job for iteration $iteration with job name: $job_name"
    echo ""
done

echo "All SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/" 