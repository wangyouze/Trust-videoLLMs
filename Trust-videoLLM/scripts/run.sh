#!/bin/bash

# Check if model_id parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

# Get the passed model_id
model_id=$1

# Set CUDA_VISIBLE_DEVICES to specify GPU (e.g., 7)
# export CUDA_VISIBLE_DEVICES="4"

# Activate the virtual environment (if needed)
# source activate MiniCPM_o_2_6

# Set the scripts directory
scripts_dir="scripts/run/truthfulness"

# Check if the directory exists
if [ ! -d "$scripts_dir" ]; then
    echo "Error: Directory $scripts_dir does not exist."
    exit 1
fi

# Loop through all .sh files in the directory and execute them with model_id
for sh_file in "$scripts_dir"/*.sh; do
    if [ -f "$sh_file" ]; then
        echo "Running script: $sh_file"
        bash "$sh_file" "$model_id"
    fi
done

echo "All scripts in $scripts_dir have been executed."