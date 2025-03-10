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
import copy
import random

from data_utils import read_json, read_txt, write_json
from expert_utils import add_brats_expert_conversation, model_list
from tqdm import tqdm

random.seed(0)

assert isinstance(model_list, str)


def main(args):
    """Prepare expert training data for brain MRI."""
    in_data = read_json(args.in_meta_data)

    assert args.n_samples < len(in_data)

    what_questions = read_txt("./llama_output/llama_gen_expert_what.txt")

    in_data = random.sample(in_data, k=args.n_samples)

    count = 0
    all_conversations = []
    for meta in tqdm(in_data, desc="creating train data..."):
        # create a q & a conversation
        entry = {
            "images": [meta["image"][0], meta["image"][1], meta["image"][2], meta["image"][3], meta["label"]],
        }

        # what question
        conv = list()
        conv.append({"from": "human", "value": random.choice(what_questions)})
        conv.append({"from": "gpt", "value": "This is a brain MRI scan."})
        entry["conversations"] = conv

        all_conversations.append(copy.copy(entry))
        count += 1

        # glioma grade question
        conv = list()
        conv.append({"from": "human", "value": "What glioma grade is shown in the image?"})
        if "LGG" in meta["orig_images"][0]:
            conv.append({"from": "gpt", "value": "The scan indicates lower grade glioma."})
        elif "HGG" in meta["orig_images"][0]:
            conv.append({"from": "gpt", "value": "The scan indicates high grade glioma."})
        else:
            raise ValueError("Could not determine glioma type")

        # add expert instructions
        new_conv = add_brats_expert_conversation(conv)

        # assert_image_placeholder(new_conv)  TODO: add check for brats image placeholders
        entry["conversations"] = new_conv

        all_conversations.append(copy.copy(entry))
        count += 1

        # Segment task
        conv = list()
        conv.append({"from": "human", "value": "Segment any tumors in the image"})
        conv.append({"from": "gpt", "value": "All tumors were segmented"})

        # add expert instructions
        new_conv = add_brats_expert_conversation(conv, trigger="I segmented any brain tumors using <BRATS()>.")

        # assert_image_placeholder(new_conv)  TODO: add check for brats image placeholders
        entry["conversations"] = new_conv

        all_conversations.append(copy.copy(entry))
        count += 1

    print(f"Converted {len(all_conversations)} conversations")

    out_train_file = args.out_fileprefix + "_train.json"
    out_test_file = args.out_fileprefix + "_test.json"

    split_idx = int(args.test_frac * len(all_conversations))

    random.shuffle(all_conversations)
    test_conversations = all_conversations[0:split_idx]
    train_conversations = all_conversations[split_idx::]

    write_json(train_conversations, out_train_file)
    write_json(test_conversations, out_test_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_meta_data", type=str, default="../../data/experts/brats/slices/extracted_slices_meta.json")
    parser.add_argument("--images_root", type=str, default="../../data/experts/brats/slices")
    parser.add_argument("--out_fileprefix", type=str, default="../../data/experts/brats/brats2018")
    parser.add_argument("--n_samples", type=int, default=10_000)
    parser.add_argument("--test_frac", type=float, default=0.5)
    args = parser.parse_args()

    main(args)
