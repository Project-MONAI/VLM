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

# load chest x-ray and try different classification prompts
import argparse
import os
import warnings
from copy import deepcopy
from types import SimpleNamespace

from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from monai.bundle.config_parser import ConfigParser
from monai.utils import look_up_option
from prompts import has_placeholder, replace, templates
from run_vila import eval_model, eval_text_model
from torchxray_cls import all_models, cls_models

cls = {1: "atelectasis", 2: "cardiomegaly", 3: "consolidation", 4: "edema", 5: "pleural effusion"}


def batch_run(exp_id, mpath, conv_mode, folder_name, p_mode="binary"):
    """Batch run for expert inference on chest x-ray images."""
    if exp_id == 0:
        p_mode = "multi_choice"  # keep the default to multi_choice mode
    print(f"using prompt_mode={p_mode}")
    prompt = look_up_option(p_mode, templates)
    if p_mode == "multi_choice":
        out_csv = os.path.join(f"{folder_name}", "test_vila_chexpert_prompt.csv")
        choices = "\n".join(f"({c}) {cls[k]}" for c, k in zip("ABCDE", cls.keys()))
        choices += "\n(F) none of the above\n"
        if not has_placeholder(prompt, "<choices>"):
            warnings.warn("Prompt template does not contain <choices> placeholder", stacklevel=2)
        prompt = replace(prompt, "<choices>", choices)
    elif p_mode in ("binary", "binary_conv", "text_only"):  # using class specific binary prompts
        out_csv = os.path.join(f"{folder_name}", f"test_vila_chexpert_{cls[exp_id].lower().replace(' ', '_')}.csv")
        if not has_placeholder(prompt, "<class_name>"):
            warnings.warn("Prompt template does not contain <class_name> placeholder", stacklevel=2)
        prompt = replace(prompt, "<class_name>", cls[exp_id])

    test_csv = "/data/datasets/chexlocalize/CheXpert/groundtruth.csv"
    image_base_dir = "/data/datasets/chexlocalize/CheXpert/"
    expert_base = "/data/datasets/chexlocalize/CheXpert/torchxrayvision/"
    expert_models = all_models
    with open(test_csv, "r") as f:
        lines = f.readlines()

    if not conv_mode:
        conv_mode = "hermes-2"
    vlm_args = SimpleNamespace(
        model_path=mpath,
        model_base=None,
        image_file="test.jpg",
        video_file=None,
        num_video_frames=6,
        query=None,
        conv_mode=conv_mode,  # "hermes-2", "llama_3", "radiology_class", "v1", "llava_v0"
        sep=",",
        temperature=0.0,
        top_p=None,
        num_beams=1,
        max_new_tokens=512,
    )
    vlm_args.model_name = get_model_name_from_path(vlm_args.model_path)
    vlm_args.tokenizer, vlm_args.model, vlm_args.image_processor, vlm_args.context_len = load_pretrained_model(
        vlm_args.model_path, vlm_args.model_name, vlm_args.model_base
    )
    try:
        os.remove(out_csv)
    except OSError:
        pass

    for idx, line in enumerate(lines[1:]):  # skip the header
        fname, label = line.strip().split(",", 1)
        fname = fname.replace("CheXpert-v1.0/", "")
        print(idx, label)
        if not has_placeholder(prompt, "<placeholder>"):
            warnings.warn("Prompt template does not contain a placeholder for the expert contents.", stacklevel=2)
            qs = deepcopy(prompt)
        else:
            ens = {}
            for expert_res in expert_models:
                json_path = os.path.join(expert_base, expert_res, fname.replace("test/", "").replace("/study1", ".json"))
                res = ConfigParser.load_config_files(json_path)
                for cls_name in res:
                    if cls_name in cls_models and expert_res not in cls_models[cls_name]:
                        continue  # this expert isn't used for this cls
                    if cls_name not in ens:
                        ens[cls_name] = [float(res[cls_name])]
                    else:
                        ens[cls_name] += [float(res[cls_name])]
            expert_results = ""
            for cls_name in ens:
                prob_cls = sum(ens[cls_name]) / len(ens[cls_name])
                expert_results += f" {cls_name.lower().replace('_', ' ')}: {'yes' if prob_cls > 0.5 else 'no'}\n"
            qs = replace(prompt, "<placeholder>", expert_results)

        conv = conv_templates[conv_mode].copy()
        for r, words in qs:
            conv.append_message(conv.roles[r], words)
        vlm_args.conv = conv

        vlm_args.image_file = os.path.join(image_base_dir, fname, "view1_frontal.jpg")
        if p_mode != "text_only":
            res = eval_model(vlm_args)
        else:
            res = eval_text_model(vlm_args)
        if res is None:
            res = ""
        res = res.replace(",", ".").replace("\n", ". ")  # comma is reserved for csv
        res_str = f"{idx},{fname},{res},{label}"
        with open(out_csv, "a") as f:
            f.write(f"{res_str}\n")


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--mpath", type=str, required=True)
    parser.add_argument("--conv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--prompt_mode", type=str, default="binary")  # which prompt template to use

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    batch_run(args.idx, args.mpath, args.conv, args.output, args.prompt_mode)
