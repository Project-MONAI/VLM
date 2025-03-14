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
import math
import os
import os.path as osp
import re
from io import BytesIO

import requests
import torch
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image


def load_filenames(file_path):
    """load_filenames"""
    with open(file_path, "r") as f:
        return [line.strip() for line in f]


def load_image(image_file):
    """load_image"""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    """load_images"""
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


def eval_model(args):
    """eval_model"""
    disable_torch_init()

    image_filenames = load_filenames(args.image_list_file)
    image_filenames = split_list(image_filenames, args.num_gpus, args.gpu_id)

    images_folder = args.images_folder
    output_folder = args.output_folder

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)

    for img_filename in image_filenames:
        image_path = osp.join(images_folder, img_filename)
        image = load_image(image_path)

        query = "Describe the image in detail."

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in query:
            if model.config.mm_use_im_start_end:
                query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
            else:
                query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
        else:
            if DEFAULT_IMAGE_TOKEN not in query:
                print(f"no <image> tag found in input. Automatically append one at the beginning of text.")
                if model.config.mm_use_im_start_end:
                    query = image_token_se + "\n" + query
                else:
                    query = DEFAULT_IMAGE_TOKEN + "\n" + query

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
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

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
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
