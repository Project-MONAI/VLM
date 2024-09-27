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
import random

from data_utils import read_json, write_json
from expert_utils import (add_expert_conversation,
                          assert_image_placeholder,
                          get_predictions,
                          model_list)
from tqdm import tqdm

random.seed(0)

assert isinstance(model_list, str)


def main(args):
    """Prepare expert training data for CXR."""
    in_data = read_json(args.in_datafile)

    assert args.n_samples < len(in_data)

    in_data = random.sample(in_data, k=args.n_samples)

    count = 0
    all_conversations = []
    for entry in tqdm(in_data, desc="creating train data..."):
        conv = entry["conversations"]

        # add expert instructions
        predictions = get_predictions(args.pred_root, entry["image"])
        new_conv = add_expert_conversation(conv, predictions)

        assert_image_placeholder(new_conv)
        entry["conversations"] = new_conv

        all_conversations.append(entry)
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
    parser.add_argument("--in_datafile", type=str, required=True)
    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--out_fileprefix", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=100_000)
    parser.add_argument("--test_frac", type=float, default=0.5)
    args = parser.parse_args()

    main(args)
