import json
import os
import re
import transformers
import torch


model_id = "Efficient-Large-Model/Meta-Llama-3.1-8B-Instruct"
# model_id = "Efficient-Large-Model/Meta-Llama-3.1-70B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

text_filename = "./sentences_test_slim.txt"
if False:
    with open(text_filename, "r") as f:
        sentences = f.read()
    print(sentences)
else:
    with open(text_filename, "r") as file:
        lines = file.readlines()
    lines = [_line.strip() for _line in lines]

N = 100
for _i in range(0, len(lines), N):
    chunk = lines[_i : _i + N]
    sentences = "\n".join(chunk)

    templates_filename = "./templates_sentences_test_slim.txt"
    templates = ""
    if os.path.exists(templates_filename):
        with open(templates_filename, "r") as file:
            templates = file.read()
    sentences += f"\n{templates}"

    messages = [
        {
            "role": "system",
            "content": "You are an expert radiologiest.",
        },
        {
            "role": "user",
            "content": f"Please simplify the following list of sentences according to these instructions: \
            1. **Break Down**: Separate each sentence into its simplest components. \
            Each resulting sentence should be straightforward and free of transitional words like 'and,' 'or,' 'but,' 'then,' 'therefore,' etc. \
            2. **Extract Similarities**: Identify sentences with similar meanings. Group these sentences based on the main idea they convey. \
            3. **Unify**: For each group of similar sentences, create a single sentence that captures the core meaning. Ensure this unified sentence is concise, clear, and without transitional words. \
            4. **Create the Final Pool**: Compile a final list of these unified, simplified sentences that represent the main content of the original list. \
            **Important**: Make sure none of the sentences include transitional words such as 'and,' 'or,' 'but,' 'then,' etc. \
            Each sentence should stand alone, conveying a single idea. \
            Here is the list of sentences to process: {sentences}. \
            Please provide the final list of simplified sentences, focusing only on the most common meanings.",
        },
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256000,
    )
    new_template = outputs[0]["generated_text"][-1]
    new_template = new_template["content"]
    print(new_template)

    with open(templates_filename, "w") as f:
        f.write(new_template)
