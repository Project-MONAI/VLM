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
import os
import pickle
import random


def process_data(data, image_prefix):
    """Process the data and return the transformed data."""
    transformed_data = []
    for item in data:
        if random.choice([True, False]):
            human_value = f"<image>\n{item['sent']}"
        else:
            human_value = f"{item['sent']}\n<image>"
        new_item = {
            "id": str(item["question_id"]),
            "image": str(os.path.join(image_prefix, f"{item['img_id']}.jpg")),
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": str(list(item["label"].keys())[0])},
            ],
        }
        transformed_data.append(new_item)
    return transformed_data


def main(args):
    """Generate the PathVQA instruct data."""
    transformed_data = []
    total_questions = 0

    # Process train data
    with open(args.train_pkl, "rb") as f:
        train_data = pickle.load(f)
    transformed_data.extend(process_data(train_data, "train"))
    total_questions += len(train_data)
    print(f"Processed {len(train_data)} train questions")

    # Process val data
    with open(args.val_pkl, "rb") as f:
        val_data = pickle.load(f)
    transformed_data.extend(process_data(val_data, "val"))
    total_questions += len(val_data)
    print(f"Processed {len(val_data)} val questions")

    # Process test data
    with open(args.test_pkl, "rb") as f:
        test_data = pickle.load(f)
    transformed_data.extend(process_data(test_data, "test"))
    total_questions += len(test_data)
    print(f"Processed {len(test_data)} test questions")

    print(f"Total questions processed: {total_questions}")

    # Write the merged JSON file
    with open(args.output_json, "w") as json_file:
        json.dump(transformed_data, json_file, indent=4)
    print(f"Merged data written to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge PathVQA instruct data")
    parser.add_argument("--train_pkl", required=True, help="Path to train_vqa.pkl")
    parser.add_argument("--val_pkl", required=True, help="Path to val_vqa.pkl")
    parser.add_argument("--test_pkl", required=True, help="Path to test_vqa.pkl")
    parser.add_argument("--output_json", required=True, help="Path to output JSON file")
    args = parser.parse_args()
    main(args)
