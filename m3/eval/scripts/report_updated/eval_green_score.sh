#!/bin/bash

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Define the input directories (refs and hyps) from the arguments
REFS_DIR=$1  # Path to the directory containing ref text files
HYPS_DIR=$2  # Path to the directory containing hyp text files
GPU_IDS=$3   # List of GPU IDs as a string, e.g., "0,1,3,4"
OUTPUT_JSON=$4  

# Create a temporary file to store the results
# RESULTS_FILE="green_scores.txt"
RESULTS_FILE="${OUTPUT_JSON}_tmp.txt"
> $RESULTS_FILE

# Convert the GPU IDs string into an array
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"

# Get the total number of GPUs specified
NUM_GPUS=${#GPU_ARRAY[@]}

# Function to run inference on a specific GPU
run_inference() {
    GPU_ID=$1
    PARTITION_INDEX=$2

    # Run the Python script on the specific GPU and partition
    CUDA_VISIBLE_DEVICES=$GPU_ID python /data/code/monai_vlm/monai_vila2d/eval/scripts/report_updated/metric_green_score.py $REFS_DIR $HYPS_DIR $NUM_GPUS $PARTITION_INDEX >> $RESULTS_FILE
}

# Loop over the number of GPUs and assign each a partition
for i in "${!GPU_ARRAY[@]}"; do
    run_inference "${GPU_ARRAY[$i]}" $i &
done

# Wait for all GPUs to finish
wait

echo "All GPU processes have finished."

# Now, compute the final average
TOTAL_SCORE=0
TOTAL_COUNT=0

# Sum up the scores and counts from all results
while read -r line; do
    # Check if the line contains exactly two numeric values (for score and count)
    if [[ $line =~ ^[0-9]+([.][0-9]+)?\ [0-9]+$ ]]; then
        SCORE=$(echo $line | cut -d ' ' -f 1)
        COUNT=$(echo $line | cut -d ' ' -f 2)
        
        # Add score and count to total
        TOTAL_SCORE=$(echo "$TOTAL_SCORE + $SCORE" | bc)
        TOTAL_COUNT=$(echo "$TOTAL_COUNT + $COUNT" | bc)
    fi
done < $RESULTS_FILE

# Compute the final average if there are valid counts
if [ $TOTAL_COUNT -gt 0 ]; then
    AVERAGE=$(echo "$TOTAL_SCORE / $TOTAL_COUNT" | bc -l | awk '{printf "%f", $0}')
    echo "Final Average GREEN Score: $AVERAGE"
    echo "{\"accuracy\": ${AVERAGE}}" > $OUTPUT_JSON
else
    echo "No valid scores to compute."
    echo "{\"accuracy\": 0}" > $OUTPUT_JSON
fi
