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
import re
import warnings

import numpy as np
from sklearn.metrics import f1_score

classes = {"Atelectasis": "a", "Cardiomegaly": "b", "Consolidation": "c", "Edema": "d", "pleural_effusion": "e"}
gt_ids = {"Atelectasis": -6, "Cardiomegaly": -12, "Consolidation": -8, "Edema": -9, "pleural_effusion": -4}
answers = {"No": 0, "no": 0, "yes": 1, "Yes": 1}


def extract_answer(text, fall_back=None):
    """Extract the answer from the text."""
    original_text = text
    text = text.strip().lower()
    if "answer: " in text:
        text = text.replace("answer:", "").strip()
    if "a: " in text:
        text = text.replace("a: ", "").strip()
    s = re.search(r"\(.*\)", text)
    if s is not None:
        text = s.group(0)[1:-1]
    s = re.search(r"(yes|no|a|b|c|d)", text)
    if s is not None:
        text = s.group(0)
    single_char = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "yes": 1, "no": 0}
    if len(text) < 1:
        if fall_back is not None:
            raise ValueError(f"unknown answer {text} -- {original_text}")
        else:
            return fall_back
    if len(text) == 1:
        if text not in single_char:
            if fall_back is not None:
                raise ValueError(f"unknown answer {text} -- {original_text}")
            else:
                return fall_back
    try:
        res = single_char[text]
    except KeyError:
        if fall_back is not None:
            print(text, original_text)
            import pdb

            pdb.set_trace()
        else:
            return fall_back
    return res


def compute_f1(args):
    """Compute the F1 score for the given answers."""
    cls_out = 0.0
    for c in classes:
        # csv_file = f"{name}/test_vila_chexpert_prompt.csv"
        csv_file = f"{args.answers}/test_vila_chexpert_{c.lower()}.csv"

        with open(csv_file, "r") as f:
            items = f.readlines()

        y_pred = []
        y_true = []
        for i in items:
            # print(i)
            the_row = i.strip().split(",")
            gt = "1" in the_row[gt_ids[c]]
            y_true.append(gt)

            pred = extract_answer(text=the_row[2], fall_back=0)
            # print(f"{the_row[2]} -- {c} -- {pred}")
            y_pred.append(pred)
        if "prompt" in os.path.basename(csv_file):  # prompt.csv means multiclass results
            y_pred = np.asarray(y_pred) == classes[c]
        elif len(np.unique(np.asarray(y_pred))) != 2:
            warnings.warn(f"not binary predictions? {csv_file}", stacklevel=2)
        try:
            out = f1_score(y_true=y_true, y_pred=y_pred, pos_label=True)
        except ValueError:  # ValueError: Target is multiclass but average='binary'.
            out = 0.0
        print(c, out)
        cls_out += out
    cls_out /= len(classes)
    print(cls_out)

    with open(args.output, "w") as f:
        json.dump({"f1": cls_out}, f)


def get_args():
    """Get arguments from command line."""
    parser = argparse.ArgumentParser()
    # parser.add_argument("--name", type=str, default="")
    # parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--answers", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    compute_f1(args)
