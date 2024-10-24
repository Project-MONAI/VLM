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
import random


def process_data(input_json, output_json, data_type):
    """Process the RadVQA data and generate the instruction data."""
    with open(input_json, "r") as j_file:
        j_data = json.load(j_file)

    # Organ list of questions (only used for training data)
    organ_q_list = [
        "What is the visible organ?",
        "What organ is it?",
        "Is any organ visible?",
        "Which can organ can be seen?",
        "Any visible organs?",
    ]

    instruct_data = []
    counter = 1

    for idx, each_data in enumerate(j_data):
        print("Status: {} / {}".format(idx + 1, len(j_data)))

        if data_type == "test":
            condition = each_data["answer_type"] == "CLOSED" and (
                each_data["phrase_type"] == "test_para" or each_data["phrase_type"] == "test_freeform"
            )
        else:  # train
            condition = each_data["phrase_type"] != "test_para" and each_data["phrase_type"] != "test_freeform"

        if condition:
            t_dict = create_conversation(each_data, counter, "question")
            instruct_data.append(t_dict)
            counter += 1
            print("Instruct Row Count: {}".format(counter))

            if data_type == "train":
                # Adding an organ question for training purposes
                t_dict = create_conversation(each_data, counter, "organ")
                instruct_data.append(t_dict)
                counter += 1
                print("Instruct Row Count: {}".format(counter))

            if "question_rephrase" in each_data and each_data["question_rephrase"] != "NULL":
                t_dict = create_conversation(each_data, counter, "question_rephrase")
                instruct_data.append(t_dict)
                counter += 1
                print("Instruct Row Count: {}".format(counter))

    with open(output_json, "w") as j_file:
        json.dump(instruct_data, j_file, indent=4)

    print("All set to fly ...")


def create_conversation(data, counter, q_type):
    """Create the conversation dictionary."""
    t_dict = {"id": str(counter), "image": data["image_name"]}

    if q_type == "organ":
        random_choice = random.randint(0, 4)
        question = organ_q_list[random_choice]
        answer = data["image_organ"]
    elif q_type == "question_rephrase":
        question = data["question_rephrase"]
        answer = data["answer"]
    else:
        question = data["question"]
        answer = data["answer"]

    t_dict["conversations"] = [
        {"from": "human", "value": str(question + "\n<image>")},
        {"from": "gpt", "value": str(answer)},
    ]
    return t_dict


def main():
    """Generate the RadVQA instruct data."""
    parser = argparse.ArgumentParser(description="Process RadVQA data for instruction tuning.")
    parser.add_argument("input_json", help="Path to the input JSON file")
    parser.add_argument("output_json", help="Path to the output JSON file")
    parser.add_argument("data_type", choices=["train", "test"], help="Type of data to process (train or test)")
    args = parser.parse_args()

    input_json = os.path.normpath(args.input_json)
    output_json = os.path.normpath(args.output_json)

    process_data(input_json, output_json, args.data_type)


if __name__ == "__main__":
    main()
