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
import os
import pprint
import sys
from glob import glob

import skimage
import skimage.io
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms
import torchxrayvision as xrv

# requires pip install scikit-image torchxrayvision
# run: python torchxray_infer.py /data/datasets/mimic-cxr/61487/images/
from monai.bundle.config_parser import ConfigParser

sys.path.insert(0, "..")
parser = argparse.ArgumentParser()
parser.add_argument("img_path", type=str)
parser.add_argument("-weights", type=str, default="densenet121-res224-all")
parser.add_argument("-feats", default=False, help="", action="store_true")
parser.add_argument("-cuda", default=False, help="", action="store_true")
parser.add_argument("-out_dir", default="/data/datasets/mimic-cxr/torchxrayvision/")

cfg = parser.parse_args()

model = xrv.models.get_model(cfg.weights)
if cfg.cuda:
    model = model.cuda()

output_full_dir = os.path.join(cfg.out_dir, cfg.weights)
if not os.path.exists(output_full_dir):
    os.makedirs(output_full_dir, exist_ok=True)

for item in glob(os.path.join(cfg.img_path, "*.jpg")):
    img = skimage.io.imread(item)
    img = xrv.datasets.normalize(img, 255)

    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    # Add color channel
    img = img[None, :, :]

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

    img = transform(img)

    output = {}
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        if cfg.cuda:
            img = img.cuda()

        if cfg.feats:
            feats = model.features(img)
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            output["feats"] = list(feats.cpu().detach().numpy().reshape(-1))

        preds = model(img).cpu()
        output["preds"] = dict(zip(xrv.datasets.default_pathologies, preds[0].detach().numpy()))
    for cls in output["preds"]:
        output["preds"][cls] = f"{output['preds'][cls]:.4f}"
    out_json = os.path.join(output_full_dir, os.path.basename(item).replace("jpg", "json"))
    ConfigParser.export_config_file(output["preds"], out_json, indent=4)

    if cfg.feats:
        print(output)
    else:
        pprint.pprint(output)
