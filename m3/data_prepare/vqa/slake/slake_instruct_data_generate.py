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
import random


# Function to randomly place "\n<image>" either before or after the question
def add_image_tag(question):
    """Add <image> tag to the question."""
    return f"{question}\n<image>" if random.choice([True, False]) else f"<image>\n{question}"


# Function to transform input data
def transform_data(input_data):
    """Transform the input data."""
    output_data = []

    for idx, item in enumerate(input_data):
        if item["q_lang"] == "en":
            transformed_item = {
                "id": str(idx + 1),
                "image": str(item["img_name"]),
                "conversations": [
                    {"from": str("human"), "value": add_image_tag(str(item["question"]))},
                    {"from": str("gpt"), "value": str(item["answer"])},
                ],
            }
            output_data.append(transformed_item)

    return output_data


def main(args):
    """Generate the Slake instruct data."""
    input_data = []
    for input_path in args.input_paths:
        with open(input_path, "r") as input_file:
            input_data.extend(json.load(input_file))

    output_data = transform_data(input_data)

    with open(args.output_path, "w") as output_file:
        json.dump(output_data, output_file, indent=4)

    print(json.dumps(output_data[:5], indent=4))  # Print first 5 items as a sample
    print(f"Total items processed: {len(output_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Slake dataset for instruction tuning")
    parser.add_argument("--input_paths", nargs="+", required=True, help="Input JSON file paths")
    parser.add_argument("--output_path", required=True, help="Output JSON file path")
    args = parser.parse_args()
    main(args)
