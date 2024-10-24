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

export CONTAINER='/path/to/container.sqsh'
export DATASETS='/path/to/datasets'
export CODE='/path/to/code'

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