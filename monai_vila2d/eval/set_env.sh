#!/bin/bash

export CONTAINER='/lustre/fsw/portfolios/healthcareeng/users/hroth/cache/image_vila_internal_pt2407.sqsh'
export DATASETS='/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/VLM/datasets'
export CODE='/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/VLM/code'

echo "Setting environment variables:"
echo "CONTAINER: $CONTAINER"
echo "DATASETS: $DATASETS"
echo "CODE: $CODE"

function print_usage {
    echo "Usage: ... checkpoint_path result_name conv_mode"
    echo "  checkpoint_path: Path to the model checkpoint"
    echo "  result_name: Name of the output folder"
    echo "  conv_mode: Convolution mode to be used"
    echo "Environment variables that can be set:"
    echo "  CONTAINER: Path to the container image file"
    echo "  DATASETS: Path to the datasets directory"
    echo "  CODE: Path to the code directory"
    exit 1
}