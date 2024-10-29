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

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


class CaptionScorer:
    def __init__(self, all_texts):
        self.scorers = [
            (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(all_texts), "Cider"),
        ]

    def __call__(self, gts, res):
        """
        Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

        :param gts: Dictionary with the image ids and their gold captions,
        :param res: Dictionary with the image ids ant their generated captions
        :print: Evaluation score (the mean of the scores of all the instances) for each measure
        """
        eval_res = {}
        for scorer, method in self.scorers:
            try:
                score, scores = scorer.compute_score(gts, res, verbose=0)
            except TypeError:
                score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    eval_res[m] = sc
            else:
                eval_res[method] = score
        return eval_res


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """
    print("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "Cider"),
    ]
    eval_res = {}

    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def read_files(directory):
    files_content = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                files_content[filename.replace(".jpg", "")] = file.read().strip()
    return files_content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", action="store", type=str, help="Ground truth directory.")
    parser.add_argument("--pred_dir", action="store", type=str, help="Prediction directory.")
    parser.add_argument("--output", action="store", type=str, help="Path to output json.")

    args = parser.parse_args()

    ground_truths = read_files(args.gt_dir)
    predictions = read_files(args.pred_dir)

    # Prepare data in COCO format
    ground_truth_data = []
    prediction_data = []

    for idx, (filename, gt_text) in enumerate(ground_truths.items()):
        if filename in predictions:
            ground_truth_data.append({"image_id": idx, "caption": gt_text})

            _text = predictions[filename]
            _text = normalize_spaces(_text.replace("\n", " "))
            prediction_data.append({"image_id": idx, "caption": normalize_spaces(_text.replace("\n", " "))})

    print(f"found {len(prediction_data)} prediction data points.")

    ground_truth_data_1 = {}
    prediction_data_1 = {}
    for _j in range(len(ground_truth_data)):
        ground_truth_data_1[_j] = [ground_truth_data[_j]]
        prediction_data_1[_j] = [prediction_data[_j]]

    eval_res = compute_scores(ground_truth_data_1, prediction_data_1)
    for key, value in eval_res.items():
        print(f"{key}: {value}")

    with open(args.output, "w") as f:
        json.dump(eval_res, f)


if __name__ == "__main__":
    main()
