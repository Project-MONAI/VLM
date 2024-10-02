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

import jsonlines


def get_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--answers", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_jsonl(file_path):
    """Load a jsonl file."""
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def extract_gpt_values(json_data):
    """Extract the GPT values from the JSON data."""
    gpt_values = {}
    for item in json_data:
        id = item["id"]
        for conversation in item["conversations"]:
            if conversation["from"] == "gpt":
                gpt_values[id] = conversation["value"].strip().lower()
    return gpt_values


def extract_text_values(jsonl_data):
    """Extract the text values from the JSONL data."""
    text_values = {}
    for item in jsonl_data:
        id = item["question_id"]
        text_values[id] = item["text"].strip().lower()
    return text_values


def calculate_accuracy(gpt_values, text_values):
    """Calculate the accuracy of the model."""
    correct = 0
    total = len(gpt_values)

    for id, gpt_value in gpt_values.items():
        if id in text_values and gpt_value == text_values[id]:
            correct += 1

    return correct / total if total > 0 else 0


def main():
    """Main function."""
    args = get_args()
    json_data = load_json(args.input)
    jsonl_data = load_jsonl(args.answers)

    gpt_values = extract_gpt_values(json_data)
    text_values = extract_text_values(jsonl_data)

    accuracy = calculate_accuracy(gpt_values, text_values)
    print(f"RADVQA {args.input} {args.answers}")
    print(f"RADVQA Accuracy: {accuracy :.4f} saved to {args.output}")

    with open(args.output, "w") as f:
        json.dump({"accuracy": accuracy}, f)


if __name__ == "__main__":
    main()
