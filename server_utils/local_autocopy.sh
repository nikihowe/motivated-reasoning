#!/bin/bash

# This script is a modified version of the original.
# It:
# - Takes a config name as a direct argument
# - Copies the motivated_reasoning directory to a temporary location 
# - Modifies the import statements in the Python files
# - Runs the experiment directly from the temporary directory

# Check if a config name is provided
if [ $# -eq 0 ]; then
    echo "Error: Config name is required"
    echo "Usage: bash server_utils/local_autocopy.sh <config_name>"
    exit 1
fi

CONFIG_NAME="${1%.*}"  # Remove .yaml extension if present

# Python file to run (should be in `training` directory)
if [ "$CONFIG_NAME" = "dummy_test" ]; then
    FILE_TO_RUN="test.py"
else
    FILE_TO_RUN="launch_training.py"
fi

# Check if /nas/ directory exists to determine if we're on the CHAI cluster
if [ -d "/nas" ]; then
    PROJ_DIR="/nas/ucb/$(whoami)/Targeted-Manipulation-and-Deception-in-LLMs"
else
    PROJ_DIR="$HOME/Targeted-Manipulation-and-Deception-in-LLMs"
fi

# Generate timestamp
TIMESTAMP=$(date +"%m_%d_%H%M%S")
TEMP_DIR="$PROJ_DIR/tmp/tmp_$TIMESTAMP"

# Check if we're already in the correct Conda environment
if [[ "$CONDA_DEFAULT_ENV" != "influence" ]]; then
    echo "Error: Not in the 'influence' Conda environment. Please activate it before running this script."
    exit 1
fi

echo "Using Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# Define the original project directory
ORIGINAL_DIR="$PROJ_DIR/motivated_reasoning"

# Create a unique temporary directory and copy the project to it
echo "Creating temporary directory: $TEMP_DIR"
mkdir -p $TEMP_DIR
cp -r $ORIGINAL_DIR $TEMP_DIR

# Modify the import statements in the tmp copy
cd $TEMP_DIR/motivated_reasoning
python utils/prep_for_slurm.py . $FILE_TO_RUN

# Run the experiment
echo "Running experiment: python training/$FILE_TO_RUN --config $CONFIG_NAME.yaml --all-gpus --timestamp $TIMESTAMP"
echo "====================CONFIG INFO===================="
python training/$FILE_TO_RUN --config $CONFIG_NAME.yaml --all-gpus --only-load-config
echo "====================END CONFIG INFO===================="
python training/$FILE_TO_RUN --config $CONFIG_NAME.yaml --all-gpus --timestamp $TIMESTAMP

# Optional: Clean up the temporary directory after the experiment finishes
# Uncomment the following line if you want to automatically delete the temporary directory
# rm -rf $TEMP_DIR