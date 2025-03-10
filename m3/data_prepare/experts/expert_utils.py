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

import os

from data_utils import read_json

model_list = (
    "Here is a list of available expert models:\n"
    "<BRATS(args)> Modality: MRI, Task: segmentation, Overview: A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data, Accuracy: Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 - Average: 0.8518, Valid args are: None\n"
    "<VISTA3D(args)> Modality: CT, Task: segmentation, Overview: domain-specialized interactive foundation model developed for segmenting and annotating human anatomies with precision, Accuracy: 127 organs: 0.792 Dice on average, Valid args are: 'everything', 'hepatic tumor', 'pancreatic tumor', 'lung tumor', 'bone lesion', 'organs', 'cardiovascular', 'gastrointestinal', 'skeleton', or 'muscles'\n"
    "<VISTA2D(args)> Modality: cell imaging, Task: segmentation, Overview: model for cell segmentation, which was trained on a variety of cell imaging outputs, including brightfield, phase-contrast, fluorescence, confocal, or electron microscopy, Accuracy: Good accuracy across several cell imaging datasets, Valid args are: None\n"
    "<CXR(args)> Modality: chest x-ray (CXR), Task: classification, Overview: pre-trained model which are trained on large cohorts of data, Accuracy: Good accuracy across several diverse chest x-rays datasets, Valid args are: None\n"
    "Give the model <NAME(args)> when selecting a suitable expert model.\n"
)
assert isinstance(model_list, str)


def assert_image_placeholder(conv):
    """Asserts that there is exactly one image placeholder in the conversation."""
    placeholder_cnt = 0
    for entry in conv:
        if "<image>" in entry["value"]:
            assert entry["from"] == "human"
            placeholder_cnt += 1

    if placeholder_cnt == 0:
        raise ValueError(f"No image placeholder found in conversation: {conv}")
    if placeholder_cnt > 1:
        raise ValueError(f"Did not expect more than one image in conversation but found {placeholder_cnt}: {conv}")


def get_predictions(root, image_name):
    """Reads the predictions from a JSON file and returns a string representation."""
    predictions_file = os.path.join(root, image_name + ".json")
    pred = read_json(predictions_file)

    pred_str = []
    for k, v in pred.items():
        if v > 0.5:
            pred_str.append(f"{k.lower().replace('_', ' ')}: yes\n")
        else:
            pred_str.append(f"{k.lower().replace('_', ' ')}: no\n")
    return "".join(pred_str)


def add_expert_conversation(conv, preds):
    """Adds expert conversation to the conversation."""
    # Keep first question
    assert conv[0]["from"] == "human"
    first_prompt = conv[0]["value"]

    new_conv = list()
    first_prompt = first_prompt.replace("\n<image>", "")
    first_prompt = first_prompt.replace("<image>", "")
    new_conv.append({"from": "human", "value": model_list + f"<image> This is a CXR image.\n" + first_prompt})
    new_conv.append({"from": "gpt", "value": "This looks like an chest x-ray. Let me first trigger <CXR()>."})
    new_conv.append(
        {
            "from": "human",
            "value": f"The resulting predictions are:\n{preds}. Analyze the image and take these predictions into account when responding to this prompt:\n{first_prompt}",
        }
    )
    new_conv.extend(conv[1::])

    return new_conv


def add_brats_expert_conversation(conv, trigger="This looks like an MRI image sequence. Let me first trigger <BRATS()>."):
    """Adds expert conversation to the conversation."""
    # Keep first question
    assert conv[0]["from"] == "human"
    first_prompt = conv[0]["value"]

    new_conv = list()
    first_prompt = first_prompt.replace("\n<image>", "")
    first_prompt = first_prompt.replace("<image>", "")
    new_conv.append(
        {
            "from": "human",
            "value": model_list
            + f"T1(contrast enhanced): <image>, T1: <image>, T2: <image>, FLAIR: <image> These are different MRI modalities.\n"
            + first_prompt,
        }
    )
    new_conv.append({"from": "gpt", "value": trigger})
    new_conv.append(
        {
            "from": "human",
            "value": f"The results are <image>. The colors in this image describe\nyellow and red: tumor core, only yellow: enhancing tumor, all colors: whole tumor\nUse this result to respond to this prompt:\n{first_prompt}.",
        }
    )
    new_conv.extend(conv[1::])

    return new_conv
