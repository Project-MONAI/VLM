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
    """Batch run for expert inference on chest x-ray images."""
    prompt = (
        "<image>\nThe following is a multiple-choice question about findings in chest X-ray in the frontal view. "
        "Please reply with the corresponding answer choice letter(s).\n"
        "Question: What are the potential abnormalities according to the provided X-ray image?\n"
        "(A) Atelectasis\n(B) Cardiomegaly\n(C) Consolidation\n(D) Edema\n(E) Pleural effusion\n(F) None of the Above\n"
    )
    prompt_0 = (
        "<image>\nThe following is a question about findings in chest X-ray in the frontal view. "
        "Please reply with yes or no.\n"
        "Question: is there atelectasis according to the provided X-ray image?\n"
    )
    prompt_1 = (
        "<image>\nThe following is a question about findings in chest X-ray in the frontal view. "
        "Please reply with yes or no.\n"
        "Question: is there cardiomegaly according to the provided X-ray image?\n"
    )
    prompt_2 = (
        "<image>\nThe following is a question about findings in chest X-ray in the frontal view. "
        "Please reply with yes or no.\n"
        "Question: is there consolidation according to the provided X-ray image?\n"
    )
    prompt_3 = (
        "<image>\nThe following is a question about findings in chest X-ray in the frontal view. "
        "Please reply with yes or no.\n"
        "Question: is there edema according to the provided X-ray image?\n"
    )
    prompt_4 = (
        "<image>\nThe following is a question about findings in chest X-ray in the frontal view. "
        "Please reply with yes or no.\n"
        "Question: is there pleural effusion according to the provided X-ray image?\n"
    )

    if not mpath.startswith("/"):
        m_path = f"Efficient-Large-Model/{mpath}"  # VILA1.5-8b
    else:
        m_path = f"{mpath}"
    # folder_name = os.path.basename(m_path.lower().replace("/checkpoint", "-iter"))
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)
    exp_set = {
        0: (prompt, f"{folder_name}/test_vila_chexpert_prompt.csv"),
        1: (prompt_0, f"{folder_name}/test_vila_chexpert_atelectasis.csv"),
        2: (prompt_1, f"{folder_name}/test_vila_chexpert_cardiomegaly.csv"),
        3: (prompt_2, f"{folder_name}/test_vila_chexpert_consolidation.csv"),
        4: (prompt_3, f"{folder_name}/test_vila_chexpert_edema.csv"),
        5: (prompt_4, f"{folder_name}/test_vila_chexpert_pleural_effusion.csv"),
    }
    out_csv = exp_set[exp_id][1]

    test_csv = "/data/datasets/chexlocalize/CheXpert/groundtruth.csv"
    image_base_dir = "/data/datasets/chexlocalize/CheXpert/"
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

    for idx, line in enumerate(lines[1:]):  # skip the header
        fname, label = line.strip().split(",", 1)
        fname = fname.replace("CheXpert-v1.0/", "")
        print(idx, label)
        vlm_args.query = exp_set[exp_id][0]
        vlm_args.image_file = os.path.join(image_base_dir, fname, "view1_frontal.jpg")
        res = eval_model(vlm_args)
        if res is None:
            res = ""
        res = res.replace(",", ".").replace("\n", ". ")  # comma is reserved for csv
        res_str = f"{idx},{fname},{res},{label}"
        with open(out_csv, "a") as f:
            f.write(f"{res_str}\n")


def get_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--mpath", type=str, required=True)
    parser.add_argument("--conv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    batch_run(args.idx, args.mpath, args.conv, args.output)
