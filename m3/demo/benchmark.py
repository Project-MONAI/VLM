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
import time
from dataclasses import dataclass
from typing import List, Tuple

import requests
import torch
import transformers
from huggingface_hub import snapshot_download
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# download image
IMAGE_URL = "https://developer.download.nvidia.com/assets/Clara/monai/samples/cxr_00030389_006.jpg"

HF_MODEL = "MONAI/Llama3-VILA-M3-8B"
CONV_MODE = "llama_3"  # Use vicuna_v1 for 3B/13B models
USER_PROMPT = "<image> Describe the image in details"
IMAGE_DIR = "."
IMAGE_PATH = "test.jpg"


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, image_path, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.image = Image.open(os.path.join(image_folder, image_path)).convert("RGB")
        self.image_tensor = process_images([self.image], self.image_processor, self.model_config)[0]

    def __getitem__(self, index):
        line = self.questions[index]
        qs = line["text"]
        image_file = line["image"]
        conv = conv_templates[CONV_MODE].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)  # or add "" to the assistant message?
        prompt = conv.get_prompt()
        if image_file == self.image_path:
            image_tensor = self.image_tensor
        else:
            raise NotImplementedError
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        return index, input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


@dataclass
class DataCollatorForVisualTextGeneration(object):
    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        indices, input_ids, images = zip(*batch)
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        images = torch.stack(images, dim=0)
        return indices, input_ids, images


def eval_model(args):
    disable_torch_init()

    model_path = snapshot_download(args.model)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name)

    # set padding side to `left` for batch text generation
    model.config.tokenizer_padding_side = tokenizer.padding_side = "left"

    questions = [{"text": args.user_prompt, "image": args.image_path}] * 12
    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    dataset = CustomDataset(questions, args.image_dir, args.image_path, tokenizer, image_processor, model.config)
    collator = DataCollatorForVisualTextGeneration(tokenizer=tokenizer)
    data_loader = DataLoader(dataset, collate_fn=collator, batch_size=1, num_workers=4, shuffle=False)
    start_time = time.time()
    l = 0
    for indices, input_ids, image_tensor in tqdm(data_loader):
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids.to(device="cuda", non_blocking=True),
                images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

            l += len(output_ids[0])

    end_time = time.time()
    print(f"Time taken to generate {l} tokens: {end_time - start_time:.2f} seconds")
    print(f"Tokens per second: {l / (end_time - start_time):.2f}")

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(f"Assistant: {outputs}")


if __name__ == "__main__":
    if not os.path.exists(os.path.join(IMAGE_DIR, IMAGE_PATH)):
        with open(os.path.join(IMAGE_DIR, IMAGE_PATH), "wb") as f:
            f.write(requests.get(IMAGE_URL).content)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=HF_MODEL)
    parser.add_argument("--conv_mode", type=str, default=CONV_MODE)
    parser.add_argument("--user_prompt", type=str, default=USER_PROMPT)
    parser.add_argument("--image_dir", type=str, default=IMAGE_DIR)
    parser.add_argument("--image_path", type=str, default=IMAGE_PATH)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()
    eval_model(args)
