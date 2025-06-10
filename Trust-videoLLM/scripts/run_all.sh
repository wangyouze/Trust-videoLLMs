#!/bin/bash

# Check if model_id parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

# Get the passed model_id
model_id=$1

# Set CUDA_VISIBLE_DEVICES to specify GPUs (4, 5, 6, 7)
export CUDA_VISIBLE_DEVICES="7"

# Activate the virtual environment
# source activate MiniCPM_o_2_6

# Set the main directory
main_dir="scripts/run/safety/"

# Loop through all subdirectories in the main directory
for sub_dir in "$main_dir"/*; do
    if [ -d "$sub_dir" ]; then
        echo "Entering directory: $sub_dir"
        # Loop through all .sh files in the subdirectory
        for sh_file in "$sub_dir"/*.sh; do
            if [ -f "$sh_file" ]; then
                echo "Running script: $sh_file"
                # Run the script file and pass model_id
                bash "$sh_file" "$model_id"
            fi
        done
    fi
done

echo "All scripts have been run"
