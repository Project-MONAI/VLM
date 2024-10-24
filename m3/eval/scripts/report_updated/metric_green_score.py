#!/usr/bin/env python

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

import os
import re
import sys

from green_score import GREEN

def load_text_files_from_directory(directory):
    """
    Load the content of all text files from a directory and return it as a list of strings.
    """
    texts = []
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith(".txt"):
            with open(filepath, "r") as file:
                content = (
                    file.read().strip()
                )  # Read the file content and remove any surrounding whitespace
                texts.append(content)
    return texts


def run_inference(refs, hyps):

    # Initialize the GREEN model (assumes correct GPU has been set in the environment)
    model = GREEN(
        model_id_or_path="StanfordAIMI/GREEN-radllama2-7b",
        do_sample=False,
        batch_size=16,
        return_0_if_no_green_score=True,
        cuda=True,
    )

    # model = GREEN("StanfordAIMI/GREEN-radllama2-7b", output_dir=".")
    # mean, std, green_score_list, summary, result_df = model(refs, hyps)

    # Initialize list to store scores
    green_scores = []

    # Calculate the GREEN score for each ref-hyp pair
    for ref, hyp in zip(refs, hyps):
        if ref.strip():  # Check if reference is not empty
            # Calculate score for the current ref-hyp pair
            score, greens, explanations = model(refs=[ref], hyps=[hyp])
            # score, std, green_score_list, summary, result_df = model(refs=[ref], hyps=[hyp])
            green_scores.append(score)

    # Calculate the sum of GREEN scores
    total_green_score = sum(green_scores)
    count = len(green_scores)

    return total_green_score, count


def partition_data(data, num_partitions, partition_index):
    """
    Partition the data into equal parts and return the part based on the partition_index.
    """
    # Compute the size of each partition
    partition_size = len(data) // num_partitions
    # Calculate start and end index for this partition
    start_index = partition_index * partition_size
    # Handle the case where it's the last partition
    if partition_index == num_partitions - 1:
        end_index = len(data)
    else:
        end_index = (partition_index + 1) * partition_size
    return data[start_index:end_index]


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def read_files(directory):
    files_content = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                files_content[filename.replace(".jpg", "")] = file.read().strip()
    return files_content


if __name__ == "__main__":
    # Get refs and hyps directories and arguments for partitioning
    refs_dir = sys.argv[1]  # refs directory path
    hyps_dir = sys.argv[2]  # hyps directory path
    num_partitions = int(sys.argv[3])  # Total number of partitions (e.g., total GPUs)
    partition_index = int(
        sys.argv[4]
    )  # Index of this partition (e.g., current GPU/process index)

    ground_truths = read_files(refs_dir)
    predictions = read_files(hyps_dir)

    refs = []
    hyps = []
    for idx, (filename, gt_text) in enumerate(ground_truths.items()):
        if filename in predictions:
            refs.append(gt_text)

            _text = predictions[filename]
            _text = normalize_spaces(_text.replace("\n", " "))
            hyps.append(normalize_spaces(_text.replace("\n", " ")))

    # Partition the refs and hyps according to the number of partitions and partition index
    refs_partition = partition_data(refs, num_partitions, partition_index)
    hyps_partition = partition_data(hyps, num_partitions, partition_index)

    # Run inference on the selected partition
    total_green_score, count = run_inference(refs_partition, hyps_partition)
    total_green_score = float(total_green_score)

    # Output the total score and count
    print(total_green_score, count)
