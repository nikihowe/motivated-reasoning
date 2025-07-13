#!/bin/bash

# Wrapper script to submit SLURM jobs for all combinations of suffix flags
# Usage: ./run_cartesian_suffixes_slurm.sh <run_name>

# Check if run name is provided
if [ $# -eq 0 ]; then
    echo "Error: Run name is required"
    echo "Usage: $0 <run_name>"
    echo "Example: $0 harmbench_kto_long_lr_5e-5-06_20_113158"
    exit 1
fi

RUN_NAME="$1"
SCRIPT_PATH="motivated_reasoning/inference/slurm/run_inference_slurm.sh"

# All combinations of the two flags
for add_true_reasoning in "" "--add_true_reasoning_suffix"; do
  for add_non_harmful in "" "--add_non_harmful_suffix"; do
    # Build a suffix for the job name for clarity
    suffix=""
    [ -n "$add_true_reasoning" ] && suffix="${suffix}_true_reasoning"
    [ -n "$add_non_harmful" ] && suffix="${suffix}_non_harmful"
    suffix="${suffix:-_no_suffix}"

    echo "Submitting jobs for RUN_NAME=$RUN_NAME with flags: $add_true_reasoning $add_non_harmful ($suffix)"
    bash $SCRIPT_PATH "$RUN_NAME" $add_true_reasoning $add_non_harmful
  done
done

echo "All cartesian product jobs submitted!" 