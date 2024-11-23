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
import re
import tempfile
from pathlib import Path
from shutil import move
from uuid import uuid4

import requests
from experts.base_expert import BaseExpert
from experts.utils import get_monai_transforms, get_slice_filenames
from monai.bundle import create_workflow


class ExpertBrats(BaseExpert):
    """Expert model for BRATS."""

    def __init__(self) -> None:
        """Initialize the VISTA-3D expert model."""
        self.model_name = "BRATS"
        self.bundle_root = os.path.expanduser("~/.cache/torch/hub/bundle/brats_mri_segmentation")

    def segmentation_to_string(
        self,
        output_dir: Path,
        img_file: str,
        seg_file: str,
        slice_index: int,
        image_filename: str,
        label_filename: str,
        modality: str = "MRI",
        axis: int = 2,
        output_prefix="The results are <segmentation>. The colors in this image describe\n",
    ):
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
        ncr = data["colormap"][1]
        # ed = data["colormap"][2]  # not used
        et = data["colormap"][4]

        return output_prefix +  f"{ncr} and {et}: tumor core, only {et}: enhancing tumor, all colors: whole tumor\n"

    def mentioned_by(self, input: str):
        """
        Check if the VISTA-3D model is mentioned in the input.

        Args:
            input (str): Text from the LLM, e.g. "Let me trigger <BRATS>."

        Returns:
            bool: True if the VISTA-3D model is mentioned, False otherwise.
        """
        matches = re.findall(r"<(.*?)>", str(input))
        if len(matches) != 1:
            return False
        return self.model_name in str(matches[0])

    def download_file(self, url: str, img_file: str):
        """
        Download the file from the URL.

        Args:
            url (str): The URL.
            img_file (str): The file path.
        """
        parent_dir = os.path.dirname(img_file)
        os.makedirs(parent_dir, exist_ok=True)
        with open(img_file, "wb") as f:
            response = requests.get(url)
            f.write(response.content)

    def run(
        self,
        img_file: list[str] | None = None,
        image_url: list[str] | None = None,
        input: str = "",
        output_dir: str = "",
        slice_index: int = 0,
        prompt: str = "",
        **kwargs,
    ):
        """
        Run the BRATS model.

        Args:
            image_url (str): The image URL list.
            input (str): The input text.
            output_dir (str): The output directory.
            img_file (str): The image file path list. If not provided, download from the URL.
            slice_index (int): The slice index.
            prompt (str): The prompt text from the original request.
            **kwargs: Additional keyword arguments.
        """
        if not img_file:
            # Download the file from the URL
            for url in image_url:
                img_file = os.path.join(output_dir, os.path.basename(url))
                self.download_file(url, img_file)

        with tempfile.TemporaryDirectory() as temp_dir:
            workflow = create_workflow(
                workflow_type="infer",
                bundle_root=self.bundle_root,
                config_file=os.path.join(self.bundle_root, f"configs/inference.json"),
                logging_file=os.path.join(self.bundle_root, "configs/logging.conf"),
                meta_file=os.path.join(self.bundle_root, "configs/metadata.json"),
                test_datalist=[{"image": img_file}],
                output_dtype="uint8",
                separate_folder=False,
                output_ext=".nii.gz",
                output_dir=temp_dir,
            )
            workflow.evaluator.run()
            output_file = os.path.join(temp_dir, os.listdir(temp_dir)[0])
            seg_file = os.path.join(output_dir, "segmentation.nii.gz")
            move(output_file, seg_file)

        seg_image = f"seg_{uuid4()}.jpg"
        text_output = self.segmentation_to_string(
            output_dir,
            img_file[0],
            seg_file,
            slice_index,
            get_slice_filenames(img_file[0], slice_index),
            seg_image,
            modality="MRI",
            axis=2,
        )

        if "segmented" in input:
            instruction = ""  # no need to ask for instruction
        else:
            instruction = "Use this result to respond to this prompt:\n" + prompt
        return text_output, os.path.join(output_dir, seg_image), instruction
