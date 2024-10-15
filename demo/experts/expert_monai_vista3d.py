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

import json
import os
import re
from io import BytesIO
from pathlib import Path
from shutil import move
from zipfile import ZipFile

import requests
from experts.base_expert import BaseExpert
from experts.utils import get_monai_transforms, get_slice_filenames


class ExpertVista3D(BaseExpert):
    """Expert model for VISTA-3D."""
    NIM_VISTA3D = "https://health.api.nvidia.com/v1/medicalimaging/nvidia/vista-3d"

    def __init__(self) -> None:
        """Initialize the VISTA-3D expert model."""
        self.model_name = "VISTA3D"

    def label_id_to_name(self, label_id: int, label_dict: dict):
        """
        Get the label name from the label ID.

        Args:
            label_id: the label ID.
            label_dict: the label dictionary.
        """
        for group_dict in list(label_dict.values()):
            if isinstance(group_dict, dict):
                # this will skip str type value, such as "everything": <path>
                for label_name, label_id_ in group_dict.items():
                    if label_id == label_id_:
                        return label_name
        return None

    def save_zipped_seg_to_file(
        self,
        zip_response: requests.Response,
        output_dir: Path,
        output_name: str = "segmentation",
        output_ext: str = ".nrrd",
    ):
        """
        Save the segmentation file from the zip response to the file.

        Args:
            zip_response: the zip response.
            output_dir: the output directory.
            output_name: the output name.
            output_ext: the output extension.
        """
        output_dir = Path(output_dir)
        if not output_dir.exists():
            os.makedirs(output_dir)
        with ZipFile(BytesIO(zip_response.content)) as zip_file:
            zip_file.extractall(output_dir)

        file_list = os.listdir(output_dir)
        for f in file_list:
            f = Path(f)
            file_path = output_dir / f
            if file_path.exists() and f.suffix == output_ext:
                move(file_path, output_dir / f"{output_name}{output_ext}")
                return output_dir / f"{output_name}{output_ext}"

        raise FileNotFoundError(f"Segmentation file not found in {output_dir}")

    def segmentation_to_string(
        self,
        output_dir: Path,
        img_file: str,
        seg_file: str,
        label_groups: dict,
        modality: str = "CT",
        slice_index: int | None = None,
        axis: int = 2,
        image_filename: str = "image.jpg",
        label_filename: str = "label.jpg",
        output_prefix="The results are <segmentation>. The colors in this image describe ",
    ):
        """
        Format the segmentation response to a string.

        Args:
            response: the response.
            output_dir: the output directory.
            img_file: the image file path.
            modality: the modality.
            slice_index: the slice index.
            axis: the axis.
            image_filename: the image filename for the sliced image.
            label_filename: the label filename for the sliced image.
            group_label_names: the group label names to filter the label names.
            output_prefix: the output prefix.
            label_groups_path: the label groups path for VISTA-3D.
        """
        output_dir = Path(output_dir)

        transforms = get_monai_transforms(
            ["image", "label"],
            output_dir,
            modality=modality,
            slice_index=slice_index,
            axis=axis,
            image_filename=image_filename,
            label_filename=label_filename,
        )
        data = transforms({"image": img_file, "label": seg_file})

        formatted_items = []

        for label_id in data["colormap"]:
            label_name = self.label_id_to_name(label_id, label_groups)
            if label_name is not None:
                color = data["colormap"][label_id]
                formatted_items.append(f"{color}: {label_name}")

        return output_prefix + ", ".join(formatted_items) + ". "

    def mentioned_by(self, input: str):
        """
        Check if the VISTA-3D model is mentioned in the input.

        Args:
            input (str): Text from the LLM, e.g. "Let me trigger <VISTA3D(arg)>."

        Returns:
            bool: True if the VISTA-3D model is mentioned, False otherwise.
        """
        return self.model_name in str(input)

    def run(
        self,
        image_url: str = "",
        input: str = "",
        output_dir: str = "",
        img_file: str = "",
        slice_index: int = 0,
        prompt: str = "",
        **kwargs,
    ):
        """
        Run the VISTA-3D model.

        Args:
            image_url (str): The image URL.
            input (str): The input text.
            output_dir (str): The output directory.
            img_file (str): The image file path.
            slice_index (int): The slice index.
            prompt (str): The prompt text from the original request.
            **kwargs: Additional keyword arguments.
        """
        output_dir = Path(output_dir)
        matches = re.findall(r"<(.*?)>", input)
        if len(matches) != 1:
            raise ValueError(f"Expert model {self.model_name} is not correctly enclosed in angle brackets.")

        match = matches[0]

        # Extract the arguments
        arg_matches = re.findall(r"\((.*?)\)", match[len(self.model_name) :])

        if len(arg_matches) == 0:  # <VISTA3D>
            arg_matches = ["everything"]
        if len(arg_matches) == 1 and (arg_matches[0] == "" or arg_matches[0] == None):  # <VISTA3D()>
            arg_matches = ["everything"]
        if len(arg_matches) > 1:
            raise ValueError(
                "Multiple expert model arguments are provided in the same prompt"
                "which is not supported in this version."
            )

        vista3d_prompts = None
        dir = os.path.dirname(__file__)
        with open(os.path.join(dir, "label_groups_dict.json")) as f:
            label_groups = json.load(f)

        if arg_matches[0] not in label_groups:
            raise ValueError(f"Label group {arg_matches[0]} not found in {label_groups}")

        if arg_matches[0] != "everything":
            vista3d_prompts = {"classes": list(label_groups[arg_matches[0]].keys())}

        # Trigger the VISTA-3D model
        api_key = os.getenv("NIM_API_KEY", "Invalid")
        if api_key == "Invalid":
            raise ValueError(f"Expert model API key not found to trigger {self.NIM_VISTA3D}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
        }

        payload = {"image": image_url}
        if vista3d_prompts is not None:
            payload["prompts"] = vista3d_prompts

        response = requests.post(self.NIM_VISTA3D, headers=headers, json=payload)
        if response.status_code != 200:
            payload.pop("image")  # hide the image URL in the error message
            raise requests.exceptions.HTTPError(
                f"Error triggering POST to {self.NIM_VISTA3D} with Payload {payload}: {response.status_code}"
            )

        seg_file = self.save_zipped_seg_to_file(response, output_dir)

        text_output = self.segmentation_to_string(
            output_dir,
            img_file,
            seg_file,
            label_groups,
            modality="CT",
            slice_index=slice_index,
            image_filename=get_slice_filenames(img_file, slice_index)[0],
            label_filename=get_slice_filenames(img_file, slice_index)[1],
        )

        if "segmented" in input:
            instruction = ""  # no need to ask for instruction
        else:
            instruction = "Use this result to respond to this prompt:\n" + prompt
        mask_overlay = os.path.join(output_dir, get_slice_filenames(img_file, slice_index)[1])
        return text_output, mask_overlay, instruction, seg_file
