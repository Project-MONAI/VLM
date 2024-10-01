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
from types import SimpleNamespace

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from run_vila import eval_model

def batch_run(exp_id, mpath, conv_mode, folder_name):
    prompt = (
        "<image>\nThe following is a multiple-choice question about findings in chest X-ray in the frontal view. "
        "Please reply with the corresponding answer choice letter(s).\n"
        "Question: What are the potential abnormalities according to the provided X-ray image?\n"
        "(A) Fracture\n(B) Pneumothorax\n(C) Lung Opacity\n(D) None of the Above\n"
    )
    prompt_0 = (
        "<image>\nThe following is a question about findings in chest X-ray in the frontal view. "
        "Please reply with yes or no.\n"
        "Question: is there fracture according to the provided X-ray image?\n"
    )
    prompt_1 = (
        "<image>\nThe following is a question about findings in chest X-ray in the frontal view. "
        "Please reply with yes or no.\n"
        "Question: is there pneumothorax according to the provided X-ray image?\n"
    )
    prompt_2 = (
        "<image>\nThe following is a question about findings in chest X-ray in the frontal view. "
        "Please reply with yes or no.\n"
        "Question: is there lung opacity according to the provided X-ray image?\n"
    )

    if not mpath.startswith("/"):
        m_path = f"Efficient-Large-Model/{mpath}"  # VILA1.5-8b
    else:
        m_path = f"{mpath}"
    # folder_name = os.path.basename(m_path.lower().replace("/checkpoint", "-iter"))
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)
    exp_set = {
        0: (prompt, f"{folder_name}/test_vila_prompt.csv"),
        1: (prompt_0, f"{folder_name}/test_vila_fracture.csv"),
        2: (prompt_1, f"{folder_name}/test_vila_pneumothorax.csv"),
        3: (prompt_2, f"{folder_name}/test_vila_lung_opacity.csv"),
    }
    out_csv = exp_set[exp_id][1]

    test_csv = "/data/datasets/chestxray14/chestxray-labels/four_findings_expert_labels/test_labels.csv"
    image_base_dir = "/data/datasets/chestxray14/83810/chestxray14_512/"
    with open(test_csv, "r") as f:
        lines = f.readlines()

    if not conv_mode:
        conv_mode = "hermes-2"
    vlm_args = SimpleNamespace(
        model_path=m_path,
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

    # print("output csv:", out_csv)
    # print("lines:", lines)

    for idx, line in enumerate(lines[1:]):  # skip the header
        fname, label = line.strip().split(",", 1)
        fname = fname.replace("png", "jpg")
        print(idx, label)
        vlm_args.query = exp_set[exp_id][0]
        vlm_args.image_file = os.path.join(image_base_dir, fname)
        res = eval_model(vlm_args)
        if res is None:
            res = ""
        res = res.replace(",", ".").replace("\n", ". ")  # comma is reserved for csv
        res_str = f"{idx},{fname},{res},{label}"
        with open(out_csv, "a") as f:
            f.write(f"{res_str}\n")
        print("witing to csv:", out_csv, res_str)

    # print("output csv:", out_csv)


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--mpath", type=str, required=True)
    parser.add_argument("--conv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    batch_run(args.idx, args.mpath, args.conv, args.output)
