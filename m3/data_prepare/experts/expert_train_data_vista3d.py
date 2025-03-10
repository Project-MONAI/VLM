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
import random
import uuid

from data_utils import read_json, read_txt, write_json
from tqdm import tqdm

random.seed(0)

from expert_utils import model_list

assert isinstance(model_list, str)

# TODO: option to upsample tumors


def get_qa_pairs(qas):
    """Get Question answer pairs from LLM generated results."""
    qa_pairs = []
    while qas.find("Q:") and qas.find("A:"):
        _start = qas.find("Q:")
        qas = qas[_start::]
        _end = qas.find("\n")
        question = qas[qas.find("Q:") + 3 : _end]
        qas = qas[_end + 1 : :]
        _end = qas.find("\n")
        answer = qas[qas.find("A:") + 3 : _end]
        qas = qas[_end::]
        # print(question, answer)
        assert "Q:" not in answer
        assert "A:" not in question
        if len(question) == 0 or len(answer) == 0:
            break
        qa_pairs.append((question, answer))

    assert len(qa_pairs) > 0
    return qa_pairs


def get_questions(reply):
    """Isolate the question."""
    questions = []
    lines = reply.split("\n")
    for line in lines:
        if len(line) > 4:
            if "." == line[1]:  # "e.g., '1.'"
                question = line[3::]
                questions.append(question)

    assert len(questions) > 0
    return questions


def parse_qas(seg_qas_raw_file, lesions_q_raw_file, how_many_q_raw_file):
    """Parse the LLM generated QA pairs."""
    seg_qas_raw = read_json(seg_qas_raw_file)
    lesions_q_raw = read_json(lesions_q_raw_file)
    how_many_q_raw = read_json(how_many_q_raw_file)

    # segmentation qa
    seg_qas = {}
    for v in seg_qas_raw:
        qas = get_qa_pairs(v["reply"])
        seg_qas[v["object_type"]] = {"reply": v["reply"], "exp_model": v["exp_model"], "qas": qas}
        print(f"Added {len(qas)} QA pairs for {v['object_type']}")

    lesions_qs = {}
    for v in lesions_q_raw:
        qs = get_questions(v["reply"])
        lesions_qs[v["tumor"]] = {"reply": v["reply"], "tumor": v["tumor"], "questions": qs}
        print(f"Added {len(qs)} lesion questions for {v['tumor']}")

    how_many_qs = {}
    for v in how_many_q_raw:
        qs = get_questions(v["reply"])
        how_many_qs[v["tumor"]] = {"reply": v["reply"], "tumor": v["tumor"], "questions": qs}
        print(f"Added {len(qs)} how many questions for {v['tumor']}")

    return seg_qas, lesions_qs, how_many_qs


def read_meta_files(root, datasets):
    """Read meta files from image extraction procedure."""
    assert isinstance(root, str)
    assert isinstance(datasets, list)
    meta_files = [os.path.join(root, ds, "extracted_slices_meta.json") for ds in datasets]

    meta = []
    out_datasets = []
    counts = []
    for meta_file, ds in zip(meta_files, datasets):
        data = read_json(meta_file)
        counts.append(len(data))
        meta.extend(data)
        out_datasets.extend(len(data) * [ds])

    print(f"Joined {counts} meta entries. Total {len(meta)}.")
    return meta, out_datasets


def find_image(images, image, dataset):
    """Find matching images."""
    assert isinstance(images, list)
    assert isinstance(image, str)
    assert isinstance(dataset, str)
    assert len(images) > 0
    for img in images:
        if dataset in img:
            if image in img:
                return img
    raise ValueError(f"Did not find a matching image for {image} and dataset {dataset}")


def main(args):
    """Prepare expert training data for VISTA3D (CT images)."""
    images = read_txt("./vista3d/ct2D_vista3d_images.txt")
    assert args.n_samples < len(images)

    incl_ds = ["Task03", "Task07", "Task09", "TotalSegmentatorV2"]
    # incl_ds = ["Task03", "Task09"]
    meta, datasets = read_meta_files(root="../experts/vista3d", datasets=incl_ds)

    # TODO: add tumor questions

    what_questions = read_txt("./llama_output/llama_gen_expert_what.txt")

    # convert raw to dict
    seg_qas, lesions_qs, how_many_qs = parse_qas(
        "./llama_output/llama_gen_expert_qa_vista3d.json",
        "./llama_output/llama_gen_expert_qa_lesions.json",
        "./llama_output/llama_gen_expert_qa_how_many.json",
    )

    meta_ds = [(m, d) for m, d in zip(meta, datasets)]
    meta_ds = random.sample(meta_ds, k=args.n_samplesn)

    all_conversations = []
    n_neg_tumors, n_pos_tumors, n_seg, n_what = 0, 0, 0, 0
    for md in tqdm(meta_ds, desc="creating train data..."):
        m, ds = md[0], md[1]
        image = find_image(images, m["image"], ds).replace(args.root_dir, "").replace("\n", "")
        label = image.replace("_img.png", "_label.png")
        group_name = m["group_name"]

        id = str(uuid.uuid4())

        entry = {"images": [image], "id": id}

        if "tumor" in group_name or "lesion" in group_name:
            # tumor task
            if group_name in lesions_qs:
                les_qs = lesions_qs[group_name]
            else:
                les_qs = lesions_qs[group_name + "s"]

            question = random.choice(les_qs["questions"])

            conv = list()
            conv.append({"from": "human", "value": model_list + f" <image>This is a CT image.\n" + question})
            conv.append({"from": "gpt", "value": f"This looks like a CT image. Let me trigger <VISTA3D({group_name})>. "})
            conv.append(
                {
                    "from": "human",
                    "value": f"The results are <image>. The colors in this image describe {m['label_colors']}. "
                    f"Use this result to respond to this prompt:\n{question}.",
                }
            )
            if len(m["num_tumors"]) > 0:
                n_pos_tumors += 1
                answer = "yes"
                conv.append({"from": "gpt", "value": answer})
                how_many = random.choice(how_many_qs["Any"]["questions"])
                conv.append({"from": "human", "value": how_many})
                answers = []
                for t, nt in zip(m["found_tumor"], m["num_tumors"]):
                    add_s = False
                    if nt > 1:
                        if not t.endswith("s"):
                            add_s = True
                    if add_s:
                        answer = f"There are {nt} {t}s."
                    else:
                        answer = f"There are {nt} {t}."
                    answers.append(answer)
                    if len(answers) > 1:
                        answers = " ".join(answers)
                    else:
                        answers = answers[0]
                conv.append({"from": "gpt", "value": answers})

                # TODO: Follow-up with what type of tumor question
            else:
                n_neg_tumors += 1
                answer = "no"
                conv.append({"from": "gpt", "value": answer})

            entry["images"].append(label)
        else:  # segmentation or what is task
            segment_task = True if random.random() > 0.5 else False
            if segment_task:
                # segmentation task
                n_seg += 1
                seg_qa = seg_qas[group_name]
                qa = random.choice(seg_qa["qas"])
                question, answer = qa[0], qa[1]

                conv = list()
                conv.append({"from": "human", "value": model_list + f" <image>This is a CT image.\n" + question})
                conv.append({"from": "gpt", "value": answer})
            else:
                n_what += 1
                question = random.choice(what_questions)
                conv = list()
                conv.append({"from": "human", "value": model_list + f" <image>This is a CT image.\n" + question})
                conv.append(
                    {"from": "gpt", "value": f"This looks like a CT image. Let me trigger <VISTA3D({group_name})>. "}
                )
                conv.append(
                    {
                        "from": "human",
                        "value": f"The results are <image>. "
                        f"The colors in this image describe {m['label_colors']}. "
                        f"Use this result to respond to this prompt:\n{question}.",
                    }
                )
                conv.append(
                    {
                        "from": "gpt",
                        "value": f"This a CT image. "
                        f"It contains several anatomical structures such as identified by VISTA3D: "
                        f"{m['label_colors']}.",
                    }
                )

        entry["conversations"] = conv

        all_conversations.append(entry)

    print(f"Converted {len(all_conversations)} conversations")

    out_train_file = args.out_fileprefix + "_train.json"
    out_test_file = args.out_fileprefix + "_test.json"

    split_idx = int(args.test_frac * len(all_conversations))

    random.shuffle(all_conversations)
    test_conversations = all_conversations[0:split_idx]
    train_conversations = all_conversations[split_idx::]

    write_json(train_conversations, out_train_file)
    write_json(test_conversations, out_test_file)

    print(
        f"Saved neg tumors: {n_neg_tumors}, pos tumors: {n_pos_tumors}, "
        f"seg: {n_seg}, what: {n_what}, total: {len(all_conversations)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--out_fileprefix", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=100_000)
    parser.add_argument("--test_frac", type=float, default=0.5)
    args = parser.parse_args()

    main(args)
