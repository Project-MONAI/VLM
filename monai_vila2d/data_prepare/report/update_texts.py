import json
import os
import random
import re
import transformers
import torch
import sys


model_id = "Efficient-Large-Model/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

input_dir = "/workspace/vlm/text_gt/dcl/train_1"
filenames = os.listdir(input_dir)
# filenames.sort()
random.shuffle(filenames)

templates_filename = "./templates_sentences_test_slim.txt"
with open(templates_filename, "r") as file:
    templates = file.read()

output_dir = f"/workspace/vlm/text_gt/dcl/train_1_update"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for _i in range(len(filenames)):
    filename = filenames[_i]

    if _i % int(sys.argv[2]) != int(sys.argv[1]):
        continue

    print(f"{_i + 1}/{len(filenames)}")

    if False:
        if os.path.exists(os.path.join(output_dir, filename)) and os.path.exists(
            os.path.join(output_dir_1, filename).replace(".jpg.txt", ".txt")
        ):
            continue
    else:
        if os.path.exists(os.path.join(output_dir, filename)):
            continue

    with open(os.path.join(input_dir, filename), "r") as file:
        report = file.read()

    messages = [
        {
            "role": "system",
            "content": "You are an expert radiologiest.",
        },
        {
            "role": "user",
            "content": f"{templates}\n\nPlease replace sentences with similar meanings in the contents below with the exact sentences from the template provided, \
            ensuring no other parts of the content are altered. Please directly output the updated report in the format 'new report: ...'.\n\n{report}",
        },
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256000,
    )
    new_report = outputs[0]["generated_text"][-1]
    new_report = new_report["content"]
    new_report = new_report.replace("new report:", "")
    new_report = new_report.replace("New report:", "")

    print(report)
    print(new_report)

    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(new_report)
