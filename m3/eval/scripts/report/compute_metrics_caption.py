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

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


# Read the files
def read_files(directory):
    """Read the files in the directory."""
    files_content = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                files_content[filename.replace(".jpg", "")] = file.read().strip()
    return files_content


def main():
    """Compute the metrics for the captioning task."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", action="store", type=str, help="Ground truth directory.")
    parser.add_argument("--pred_dir", action="store", type=str, help="Prediction directory.")
    parser.add_argument("--output", type=str, required=True, help="Path to output json.")

    args = parser.parse_args()

    ground_truths = read_files(args.gt_dir)
    predictions = read_files(args.pred_dir)

    # Prepare data in COCO format
    ground_truth_data = []
    prediction_data = []

    for idx, (filename, gt_text) in enumerate(ground_truths.items()):
        ground_truth_data.append({"image_id": idx, "caption": gt_text})
        # filename = filename.replace(".jpg", "")
        # print(filename)
        if filename in predictions:
            prediction_data.append({"image_id": idx, "caption": predictions[filename]})
        else:
            print(f"Warning: No prediction for {filename}")

    print(f"found {len(prediction_data)} prediction data points.")

    # Save ground truths in COCO format
    annotation_file = "annotations.json"
    with open(annotation_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "info": {},
                "images": [{"id": i} for i in range(len(ground_truth_data))],
                "annotations": [
                    {"image_id": d["image_id"], "caption": d["caption"], "id": i} for i, d in enumerate(ground_truth_data)
                ],
                "licenses": [],
                "type": "captions",
            },
            f,
        )

    # Save predictions in COCO result format
    results_file = "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            [{"image_id": d["image_id"], "caption": d["caption"]} for d in prediction_data],
            f,
        )

    # Create COCO object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # Create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # Evaluate on a subset of images by setting
    coco_eval.params["image_id"] = coco_result.getImgIds()

    # Evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # Print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    with open(args.output, "w") as f:
        json.dump({"accuracy": coco_eval.eval}, f)


if __name__ == "__main__":
    main()
