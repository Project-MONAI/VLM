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
import json
import pickle


def get_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--answers", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


def load_pkl_file(file_path):
    """Load a pickle file."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_jsonl_file(file_path):
    """Load a jsonl file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def calculate_accuracy(pkl_data, jsonl_data):
    """Calculate the accuracy of the model."""
    yes_no_data = [item for item in pkl_data if item.get("answer_type") == "yes/no"]

    jsonl_dict = {item["question_id"]: item["text"].lower() for item in jsonl_data}

    correct_predictions = 0
    total_predictions = len(yes_no_data)

    for item in yes_no_data:
        question_id = item["question_id"]
        label = list(item["label"].keys())[0].lower()

        # print(jsonl_dict.get(str(question_id)))
        if jsonl_dict.get(str(question_id)) == label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy


def main():
    """Main function."""
    args = get_args()
    pkl_data = load_pkl_file(args.input)
    jsonl_data = load_jsonl_file(args.answers)

    accuracy = calculate_accuracy(pkl_data, jsonl_data)
    print(f"PATHVQA {args.input} {args.answers}")
    print(f"PATHVQA Accuracy: {accuracy :.4f} saved to {args.output}")

    with open(args.output, "w") as f:
        json.dump({"accuracy": accuracy}, f)


if __name__ == "__main__":
    main()
