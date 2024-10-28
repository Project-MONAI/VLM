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

import argparse
import csv
import json
import os
import random


def process_data_from_csv(file_path, image_prefix):
    """Process the data from a CSV file and return the transformed data."""
    transformed_data = []
    with open(file_path, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if random.choice([True, False]):
                human_value = f"<image>\n{row['question']}"
            else:
                human_value = f"{row['question']}\n<image>"
            new_item = {
                "id": row["file_name"],
                "image": str(os.path.join(image_prefix, f"{row['file_name']}")),
                "conversations": [
                    {"from": "human", "value": human_value},
                    {"from": "gpt", "value": row["answer"]},
                ],
            }
            transformed_data.append(new_item)
    return transformed_data


def main(args):
    """Generate the PathVQA instruct data."""
    train_val_data = []
    test_data = []
    total_questions = 0

    # Define file paths
    train_csv_path = os.path.join(args.input_dir, "train_metadata.csv")
    val_csv_path = os.path.join(args.input_dir, "val_metadata.csv")
    test_csv_path = os.path.join(args.input_dir, "test_metadata.csv")

    # Process train data
    train_data = process_data_from_csv(train_csv_path, "train")
    train_val_data.extend(train_data)
    total_questions += len(train_data)
    print(f"Processed {len(train_data)} train questions")

    # Process val data
    val_data = process_data_from_csv(val_csv_path, "val")
    train_val_data.extend(val_data)
    total_questions += len(val_data)
    print(f"Processed {len(val_data)} val questions")

    # Process test data
    test_data = process_data_from_csv(test_csv_path, "test")
    total_questions += len(test_data)
    print(f"Processed {len(test_data)} test questions")

    print(f"Total questions processed: {total_questions}")

    # Define output file paths
    train_val_json_path = os.path.join(args.output_dir, "train_instruct.json")
    test_json_path = os.path.join(args.output_dir, "test_instruct.json")

    # Write the train and validation JSON file
    with open(train_val_json_path, "w") as json_file:
        json.dump(train_val_data, json_file, indent=4)
    print(f"Train and validation data written to {train_val_json_path}")

    # Write the test JSON file
    with open(test_json_path, "w") as json_file:
        json.dump(test_data, json_file, indent=4)
    print(f"Test data written to {test_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge PathVQA instruct data")
    parser.add_argument(
        "--input_dir", default="set/path/to/csv/files", required=False, help="Directory containing the CSV files"
    )
    parser.add_argument(
        "--output_dir",
        default="set/path/for/output/json/files",
        required=False,
        help="Directory to output the JSON files",
    )
    args = parser.parse_args()
    main(args)
