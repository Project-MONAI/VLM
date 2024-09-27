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

import base64
import os
import pickle


def encode_image_to_base64(image_path):
    """Encode the 2D image to base64 string."""
    with open(image_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode("utf-8")
    return base64_string


def save_dataset(dataset_type, dataset_name, save_path, data):
    """Save the dataset to a pickle file."""
    save_filename = f"{dataset_type}_{dataset_name}.pkl"
    save_pathname = os.path.join(save_path, save_filename)
    with open(save_pathname, "wb") as f:
        pickle.dump(data, f)


image_dir = "./images"
output_dir = "./gt"

list_filepath = "./list.txt"
text_dir = "./text_gt"

with open(list_filepath, "r") as file:
    filepaths = file.readlines()
filepaths = [_item.strip() for _item in filepaths]
num_cases = len(filepaths)

data_dict = []
for _i in range(num_cases):
    print(f"{_i + 1}/{num_cases}")
    text_filepath = os.path.join(text_dir, filepaths[_i])
    image_filepath = os.path.join(image_dir, filepaths[_i])
    image_filepath = image_filepath.replace(".txt", "")
    print(image_filepath, text_filepath)

    if not os.path.exists(image_filepath):
        continue

    image_base64_str = encode_image_to_base64(image_filepath)
    with open(text_filepath, "r") as file:
        reference_report = file.read()

    _dict = {
        "question": "Describe the image in detail.",
        "image": [image_base64_str],
        "answer": reference_report,
    }
    data_dict.append(_dict)

save_dataset("captioning", "mimic_train", output_dir, data_dict)
