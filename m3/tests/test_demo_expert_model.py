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

import sys
import unittest
from dotenv import load_dotenv
import tempfile

load_dotenv()
import os

sys.path.append("demo/experts")

from expert_torchxrayvision import ExpertTXRV
from expert_monai_vista3d import ExpertVista3D
from utils import get_slice_filenames, get_monai_transforms, save_image_url_to_file

VISTA_URL = "https://developer.download.nvidia.com/assets/Clara/monai/samples/liver_0.nii.gz"
CXR_URL = "https://developer.download.nvidia.com/assets/Clara/monai/samples/cxr_ce3d3d98-bf5170fa-8e962da1-97422442-6653c48a_v1.jpg"


class TestExperts(unittest.TestCase):
    def test_run_vista3d(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt = "This seems a CT image. Let me trigger <VISTA3D(everything)>."
            vista3d = ExpertVista3D()
            self.assertTrue(vista3d.mentioned_by(prompt))
            img_file = save_image_url_to_file(VISTA_URL, temp_dir)
            output_text, seg_file, _ = vista3d.run(
                image_url=VISTA_URL,
                input=prompt,
                output_dir=temp_dir,
                img_file=img_file,
                slice_index=0,
                prompt="",
            )

            self.assertTrue(output_text is not None)
            self.assertTrue(os.path.exists(seg_file))

    def test_run_vista3d_no_followup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt = "I segmented the image with <VISTA3D(everything)>."
            vista3d = ExpertVista3D()
            self.assertTrue(vista3d.mentioned_by(prompt))
            img_file = save_image_url_to_file(VISTA_URL, temp_dir)
            _, _, instruction = vista3d.run(
                image_url=VISTA_URL,
                input=prompt,
                output_dir=temp_dir,
                img_file=img_file,
                slice_index=0,
                prompt="",
            )

            self.assertTrue(instruction == "")

    def test_run_cxr(self):
        input = "This seems a CXR image. Let me trigger <CXR>."
        cxr = ExpertTXRV()
        self.assertTrue(cxr.mentioned_by(input))

        output_text, file, _ = cxr.run(image_url=CXR_URL, prompt="")
        print(output_text)
        self.assertTrue(output_text is not None)
        self.assertTrue(file is None)


class TestExpertUtils(unittest.TestCase):
    def test_filename_slices(self):
        filename = "data/ct_image.nii.gz"
        img  = get_slice_filenames(filename, 0)
        self.assertEqual(img, "ct_image_slice0.jpg")

    def test_monai_transforms(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # Download the TEST_IMAGE_3D
            img_file = save_image_url_to_file(VISTA_URL, tempdir)
            image_filename = "slice0.jpg"
            compose = get_monai_transforms(
                ["image"], tempdir, modality="CT", slice_index=0, image_filename=image_filename
            )
            compose({"image": img_file})
            self.assertEqual(os.path.exists(os.path.join(tempdir, image_filename)), True)


if __name__ == "__main__":
    unittest.main()
