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


def get_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=True)
    return parser.parse_args()


def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    """Print the metrics."""
    args = get_args()
    results = {}

    try:
        json_data = load_json(os.path.join(args.input, "radvqa/results.json"))
        results["radvqa"] = json_data["accuracy"]
    except:
        results["radvqa"] = "?"

    try:
        json_data = load_json(os.path.join(args.input, "mimicvqa/results.json"))
        results["mimicvqa"] = json_data["wavg_closed"]
    except:
        results["mimicvqa"] = "?"

    try:
        json_data = load_json(os.path.join(args.input, "slakevqa/results.json"))
        results["slakevqa"] = json_data["accuracy"]
    except:
        results["slakevqa"] = "?"

    try:
        json_data = load_json(os.path.join(args.input, "pathvqa/results.json"))
        results["pathvqa"] = json_data["accuracy"]
    except:
        results["pathvqa"] = "?"

    try:
        json_data = load_json(os.path.join(args.input, "report_mimiccxr/results.json"))
        results["report_mimiccxr_old_bleu4"] = json_data["BLEU_4"]
        results["report_mimiccxr_old_rougel"] = json_data["ROUGE_L"]

    except:
        results["report_mimiccxr_old_bleu4"] = "?"
        results["report_mimiccxr_old_rougel"] = "?"

    try:
        json_data = load_json(os.path.join(args.input, "report_mimiccxr/result_green.json"))
        results["report_mimiccxr_old_green"] = json_data["accuracy"]
    except:
        results["report_mimiccxr_old_green"] = "?"

    try:
        json_data = load_json(os.path.join(args.input, "report_mimiccxr/results_clean.json"))
        results["report_mimiccxr_clean_bleu4"] = json_data["BLEU_4"]
        results["report_mimiccxr_clean_rougel"] = json_data["ROUGE_L"]

    except:
        results["report_mimiccxr_clean_bleu4"] = "?"
        results["report_mimiccxr_clean_rougel"] = "?"

    try:
        json_data = load_json(os.path.join(args.input, "report_mimiccxr/result_clean_green.json"))
        results["report_mimiccxr_clean_green"] = json_data["accuracy"]
    except:
        results["report_mimiccxr_clean_green"] = "?"

    try:
        json_data = load_json(os.path.join(args.input, "report_mimiccxr/results_clean_expert.json"))
        results["report_mimiccxr_clean_expert_bleu4"] = json_data["BLEU_4"]
        results["report_mimiccxr_clean_expert_rougel"] = json_data["ROUGE_L"]

    except:
        results["report_mimiccxr_clean_expert_bleu4"] = "?"
        results["report_mimiccxr_clean_expert_rougel"] = "?"

    try:
        json_data = load_json(os.path.join(args.input, "report_mimiccxr/result_clean_expert_green.json"))
        results["report_mimiccxr_clean_expert_green"] = json_data["accuracy"]
    except:
        results["report_mimiccxr_clean_expert_green"] = "?"

    try:
        json_data = load_json(os.path.join(args.input, "chestxray14_class/results.json"))
        results["chestxray14_class"] = json_data["f1"]
    except:
        results["chestxray14_class"] = "?"

    try:
        json_data = load_json(os.path.join(args.input, "chexpert_class/results.json"))
        results["chexpert_class"] = json_data["f1"]
    except:
        results["chexpert_class"] = "?"

    try:
        json_data = load_json(os.path.join(args.input, "chestxray14_expert_class/results.json"))
        results["chestxray14_expert_class"] = json_data["f1"]
    except:
        results["chestxray14_expert_class"] = "?"

    try:
        json_data = load_json(os.path.join(args.input, "chexpert_expert_class/results.json"))
        results["chexpert_expert_class"] = json_data["f1"]
    except:
        results["chexpert_expert_class"] = "?"

    results = json.loads(json.dumps(results), parse_float=lambda x: round(100 * float(x), 2))  # convert to percentage

    print(json.dumps(results, indent=4))
    print([v for _, v in results.items()])


if __name__ == "__main__":
    main()
