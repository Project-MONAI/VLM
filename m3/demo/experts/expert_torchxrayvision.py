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

"""
The implementation of TorchXRayVision expert model is adapted from the Get-Started example in TorchXRayVision:
https://github.com/mlmed/torchxrayvision
"""

import re

import skimage.io
import torch
import torchxrayvision as xrv
from experts.base_expert import BaseExpert

MODEL_NAMES = [
    "densenet121-res224-all",
    "densenet121-res224-chex",
    "densenet121-res224-mimic_ch",
    "densenet121-res224-mimic_nb",
    "densenet121-res224-nih",
    "densenet121-res224-pc",
    "densenet121-res224-rsna",
    "resnet50-res512-all",
]

# Taken from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/models.py
valid_labels_model = {
    "densenet121-res224-all": [
        "Atelectasis",
        "Consolidation",
        "Infiltration",
        "Pneumothorax",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Effusion",
        "Pneumonia",
        "Pleural_Thickening",
        "Cardiomegaly",
        "Nodule",
        "Mass",
        "Hernia",
        "Lung Lesion",
        "Fracture",
        "Lung Opacity",
        "Enlarged Cardiomediastinum",
    ],
    "densenet121-res224-chex": [
        "Atelectasis",
        "Consolidation",
        "",
        "Pneumothorax",
        "Edema",
        "",
        "",
        "Effusion",
        "Pneumonia",
        "",
        "Cardiomegaly",
        "",
        "",
        "",
        "Lung Lesion",
        "Fracture",
        "Lung Opacity",
        "Enlarged Cardiomediastinum",
    ],
    "densenet121-res224-mimic_ch": [
        "Atelectasis",
        "Consolidation",
        "",
        "Pneumothorax",
        "Edema",
        "",
        "",
        "Effusion",
        "Pneumonia",
        "",
        "Cardiomegaly",
        "",
        "",
        "",
        "Lung Lesion",
        "Fracture",
        "Lung Opacity",
        "Enlarged Cardiomediastinum",
    ],
    "densenet121-res224-mimic_nb": [
        "Atelectasis",
        "Consolidation",
        "",
        "Pneumothorax",
        "Edema",
        "",
        "",
        "Effusion",
        "Pneumonia",
        "",
        "Cardiomegaly",
        "",
        "",
        "",
        "Lung Lesion",
        "Fracture",
        "Lung Opacity",
        "Enlarged Cardiomediastinum",
    ],
    "densenet121-res224-nih": [
        "Atelectasis",
        "Consolidation",
        "Infiltration",
        "Pneumothorax",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Effusion",
        "Pneumonia",
        "Pleural_Thickening",
        "Cardiomegaly",
        "Nodule",
        "Mass",
        "Hernia",
        "",
        "",
        "",
        "",
    ],
    "densenet121-res224-pc": [
        "Atelectasis",
        "Consolidation",
        "Infiltration",
        "Pneumothorax",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Effusion",
        "Pneumonia",
        "Pleural_Thickening",
        "Cardiomegaly",
        "Nodule",
        "Mass",
        "Hernia",
        "",
        "Fracture",
        "",
        "",
    ],
    "densenet121-res224-rsna": [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "Pneumonia",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "Lung Opacity",
        "",
    ],
    "resnet50-res512-all": [
        "Atelectasis",
        "Consolidation",
        "Infiltration",
        "Pneumothorax",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Effusion",
        "Pneumonia",
        "Pleural_Thickening",
        "Cardiomegaly",
        "Nodule",
        "Mass",
        "Hernia",
        "Lung Lesion",
        "Fracture",
        "Lung Opacity",
        "Enlarged Cardiomediastinum",
    ],
}


# Copied from https://github.com/Project-MONAI/VLM/blob/b74d7e444c6604ea83846b67bb98b5172a7e5495/m3/data_prepare/experts/torchxrayvision/torchxray_cls.py#L47-L54
group_0 = ["resnet50-res512-all"]
group_1 = [
    "densenet121-res224-all",
    "densenet121-res224-nih",
    "densenet121-res224-chex",
    "densenet121-res224-mimic_ch",
    "densenet121-res224-mimic_nb",
    "densenet121-res224-rsna",
    "densenet121-res224-pc",
    "resnet50-res512-all",
]
group_2 = [
    "densenet121-res224-all",
    "densenet121-res224-chex",
    "densenet121-res224-pc",
    "resnet50-res512-all",
]
group_4 = [
    "densenet121-res224-all",
    "densenet121-res224-nih",
    "densenet121-res224-chex",
    "resnet50-res512-all",
]

cls_models = {
    "Fracture": group_4,
    "Pneumothorax": group_0,
    "Lung Opacity": group_1,
    "Atelectasis": group_2,
    "Cardiomegaly": group_2,
    "Consolidation": group_2,
    "Edema": group_2,
    "Effusion": group_2,
}


class ExpertTXRV(BaseExpert):
    """Expert model for the TorchXRayVision model."""

    def __init__(self) -> None:
        """Initialize the CXR expert model."""
        self.model_name = "CXR"
        self.models = {}
        for name in MODEL_NAMES:
            if "densenet" in name:
                self.models[name] = xrv.models.DenseNet(weights=name).to("cuda")
            elif "resnet" in name:
                self.models[name] = xrv.models.ResNet(weights=name).to("cuda")

    def classification_to_string(self, outputs):
        """Format the classification outputs to a string."""

        def binary_output(value):
            return "yes" if value >= 0.5 else "no"

        def score_output(value):
            return f"{value:.2f}"

        formatted_items = [f"{key.lower().replace('_', ' ')}: {binary_output(outputs[key])}" for key in sorted(outputs)]

        return "\n".join(["The resulting predictions are:"] + formatted_items + ["."])

    def mentioned_by(self, input: str):
        """
        Check if the CXR model is mentioned in the input string.

        Args:
            input (str): Text from the LLM, e.g. "Let me trigger <CXR>."

        Returns:
            bool: True if the CXR model is mentioned, False otherwise.
        """
        matches = re.findall(r"<(.*?)>", str(input))
        if len(matches) != 1:
            return False
        return self.model_name in str(matches[0])

    def run(self, image_url: str = "", prompt: str = "", **kwargs):
        """
        Run the CXR model to classify the image.

        Args:
            image_url (str): The image URL.
            prompt: the original prompt.

        Returns:
            tuple: The classification string, file path, and the next step instruction.
        """

        img = skimage.io.imread(image_url)
        img = xrv.datasets.normalize(img, 255)

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        elif len(img.shape) < 2:
            raise ValueError("error, dimension lower than 2 for image")  # FIX TBD

        # Add color channel
        img = img[None, :, :]
        img = xrv.datasets.XRayCenterCrop()(img)

        preds_label = {label: [] for label in xrv.datasets.default_pathologies}

        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0).to("cuda")
            for name, model in self.models.items():
                preds = model(img).cpu()
                for k, v in zip(xrv.datasets.default_pathologies, preds[0].detach().numpy()):
                    # TODO: Exclude invalid labels, which may be different from training
                    # if k not in valid_labels_model[name]:
                    #     continue
                    # skip if k is one of the 8 pre-selected classes but the model isn't in the group
                    if k in cls_models and name not in cls_models[k]:
                        continue
                    preds_label[k].append(float(v))
            output = {k: float(sum(v) / len(v)) for k, v in preds_label.items()}
        return (
            self.classification_to_string(output),
            None,
            "Use this result to respond to this prompt:\n" + prompt,
        )
