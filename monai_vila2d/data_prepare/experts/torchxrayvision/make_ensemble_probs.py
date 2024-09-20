import os
from glob import glob

from monai.bundle.config_parser import ConfigParser

from torchxray_cls import all_models, cls_models

input_dir = "/data/datasets/mimic-cxr/torchxrayvision/"
output_dir = "/data/datasets/mimic-cxr/torchxrayvision/ensemble/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

files = glob(os.path.join(input_dir, "densenet121-res224-all", "*.json"))

for idx, item in enumerate(files):
    item_id = item.replace(input_dir + "densenet121-res224-all/", "")
    out_json = os.path.join(output_dir, item_id.replace(".json", ".jpg.json"))
    print(f"({idx + 1}/{len(files)}) writing {out_json}")

    if os.path.exists(out_json):
        continue

    ens = {}
    for expert_res in all_models:
        json_path = os.path.join(input_dir, expert_res, item_id)
        res = ConfigParser.load_config_files(json_path)
        for cls_name in res:
            if cls_name in cls_models and expert_res not in cls_models[cls_name]:
                continue
            if cls_name not in ens:
                ens[cls_name] = [float(res[cls_name])]
            else:
                ens[cls_name] += [float(res[cls_name])]
    for cls_name in ens:
        ens[cls_name] = sum(ens[cls_name]) / len(ens[cls_name])
    ConfigParser.export_config_file(ens, out_json, indent=4)


files = glob(os.path.join(output_dir, "*.json"))

for idx, item in enumerate(files):
    res = ConfigParser.load_config_files(item)
    print(f"{idx + 1} of {len(files)}")
    if len(res) != 18:
        raise ValueError(f"output item {len(res)} is not correct {res}")
