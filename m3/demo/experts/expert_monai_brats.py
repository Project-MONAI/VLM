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
import sys
import tempfile
from pathlib import Path
from shutil import move
from uuid import uuid4

import requests
from experts.utils import get_monai_transforms, get_slice_filenames, save_image_url_to_file
from monai.bundle import create_workflow

class ExpertBrats():
    def __init__(self) -> None:
        """Initialize the VISTA-3D expert model."""
        self.model_name = "BRATS"
        self.bundle_root = os.path.expanduser("~/.cache/torch/hub/bundle/brats_mri_segmentation")

    def download_file(self, url: str, img_file: str):
        """
        Download the file from the URL.

        Args:
            url (str): The URL.
            img_file (str): The file path.
        """


    def run(
            self,
            image_url: list[str] | None = None,
            img_file: list[str] | None = None,
            output_dir: str = ".",
    ):
        """Run the model"""

        if not img_file:
            # Download the file from the URL
            for url in image_url:
                img_file = os.path.join(output_dir, os.path.basename(url))
                self.download_file(url, img_file)



if __name__ == "__main__":
    expert = ExpertBrats()
    expert.run(image_url=[
        "https://developer.download.nvidia.com/assets/Clara/monai/samples/mri_Brats18_2013_31_1_t1.nii.gz",
        "https://developer.download.nvidia.com/assets/Clara/monai/samples/mri_Brats18_2013_31_1_t1ce.nii.gz",
        "https://developer.download.nvidia.com/assets/Clara/monai/samples/mri_Brats18_2013_31_1_t2.nii.gz",
        "https://developer.download.nvidia.com/assets/Clara/monai/samples/mri_Brats18_2013_31_1_flair.nii.gz",
    ], output_dir="/tmp/brats")
