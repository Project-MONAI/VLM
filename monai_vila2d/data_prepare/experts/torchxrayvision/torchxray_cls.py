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

all_models = [
    "densenet121-res224-all",
    "densenet121-res224-nih",
    "densenet121-res224-chex",
    "densenet121-res224-mimic_ch",
    "densenet121-res224-mimic_nb",
    "densenet121-res224-rsna",
    "densenet121-res224-pc",
    "resnet50-res512-all",
]
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
