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
import re
from io import BytesIO
import os
import os.path as osp

import json
import math
import requests
import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


model_list = (
    "Here is a list of available expert models:\n"
    "<BRATS(args)> Modality: MRI, Task: segmentation, Overview: A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data, Accuracy: Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 - Average: 0.8518, Valid args are: None\n"
    "<VISTA3D(args)> Modality: CT, Task: segmentation, Overview: domain-specialized interactive foundation model developed for segmenting and annotating human anatomies with precision, Accuracy: 127 organs: 0.792 Dice on average, Valid args are: 'everything', 'hepatic tumor', 'pancreatic tumor', 'lung tumor', 'bone lesion', 'organs', 'cardiovascular', 'gastrointestinal', 'skeleton', or 'muscles'\n"
    "<VISTA2D(args)> Modality: cell imaging, Task: segmentation, Overview: model for cell segmentation, which was trained on a variety of cell imaging outputs, including brightfield, phase-contrast, fluorescence, confocal, or electron microscopy, Accuracy: Good accuracy across several cell imaging datasets, Valid args are: None\n"
    "<CXR(args)> Modality: chest x-ray (CXR), Task: classification, Overview: pre-trained model which are trained on large cohorts of data, Accuracy: Good accuracy across several diverse chest x-rays datasets, Valid args are: None\n"
    "Give the model <NAME(args)> when selecting a suitable expert model.\n"
)
assert isinstance(model_list, str)


def load_filenames(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f]


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    return [load_image(image_file) for image_file in image_files]


def split_list(filenames, num_gpus, gpu_rank):
    """
    Splits the list of filenames evenly across the number of GPUs.
    Returns only the portion assigned to the current GPU (by rank).
    """
    total_files = len(filenames)
    files_per_gpu = math.ceil(total_files / num_gpus)

    start_idx = gpu_rank * files_per_gpu
    end_idx = min(start_idx + files_per_gpu, total_files)

    return filenames[start_idx:end_idx]


def get_predictions(root, image_name):
    """Reads the predictions from a JSON file and returns a string representation."""
    predictions_file = os.path.join(root, image_name + ".json")
    # pred = read_json(predictions_file)
    with open(predictions_file, "r") as f:
        pred = json.load(f)

    pred_str = []
    for k, v in pred.items():
        if v > 0.5:
            pred_str.append(f"{k.lower().replace('_', ' ')}: yes\n")
        else:
            pred_str.append(f"{k.lower().replace('_', ' ')}: no\n")
    return "".join(pred_str)


def eval_model(args):
    disable_torch_init()

    image_filenames = load_filenames(args.image_list_file)
    image_filenames = split_list(image_filenames, args.num_gpus, args.gpu_id)

    images_folder = args.images_folder
    output_folder = args.output_folder

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, model_name, args.model_base
    )

    for img_filename in image_filenames:
        image_path = osp.join(images_folder, img_filename)
        image = load_image(image_path)

        preds = get_predictions(args.expert_model_folder, img_filename)

        query = "Describe the image in detail."

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        print(f"args.conv_mode: {args.conv_mode}")

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], model_list + f"<image> This is a CXR image.\n" + query)
        conv.append_message(conv.roles[1], "This looks like an chest x-ray. Let me first trigger <CXR()>.")
        conv.append_message(
            conv.roles[0],
            f"The resulting predictions are:\n{preds}. Analyze the image and take these predictions "
            f"into account when responding to this prompt.\n{query}"
        )
        conv.append_message(conv.roles[1], None)
        print(conv)
        prompt = conv.get_prompt()

        images_tensor = process_images([image], image_processor, model.config).to(
            model.device, dtype=torch.float16
        )
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[
                    images_tensor,
                ],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        print(f"\noutputs: {outputs}\n")

        # Save output to a file
        output_filename = osp.splitext(img_filename)[0] + ".txt"
        output_path = osp.join(output_folder, output_filename)
        with open(output_path, "w") as f:
            f.write(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Model/8b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-list-file", type=str, required=True)
    parser.add_argument("--images-folder", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--expert-model-folder", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
