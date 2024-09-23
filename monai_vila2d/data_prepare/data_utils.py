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

import json
import os

# from https://github.com/scikit-image/scikit-image/blob/fe57b47ba46e9e11c626b97a8a24cd4d2c41fba1/skimage/color/colorlabel.py#L14
colors = (
    'red',
    'blue',
    'yellow',
    'magenta',
    'green',
    'indigo',
    'darkorange',
    'cyan',
    'pink',
    'brown',
    'orange',
    'lime',
    'orange',
    'gold',
    'yellowgreen',
    'darkgreen',

)


def read_txt(filename, ignore_newline=True):
    assert ".txt" in filename
    with open(filename, "r") as f:
        data = f.readlines()
    if ignore_newline:
        data = [d.replace("\n", "") for d in data]
    return data


def read_json(filename):
    assert ".json" in filename
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def read_jsonl(filename):
    assert ".jsonl" in filename
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data


def write_json(data, out_filename, indent=4):
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    with open(out_filename, "w") as f:
        json.dump(data, f, indent=indent)
    print(f"Wrote {len(data)} entries to {out_filename}")


def remove_extension(filename):
    out_filename = filename
    while len(os.path.splitext(out_filename)[1]) > 0:
        out_filename = os.path.splitext(out_filename)[0]
    return out_filename


def listdir(path, ignore_hidden=True):
    if ignore_hidden:
        dirs = []
        for f in os.listdir(path):
            if not f.startswith('.'):
                dirs.append(f)
        return dirs
    else:
        return os.listdir(path)

def get_label_name(label_dict, search_id):
    assert search_id > 0
    found_name = None
    for name, _id in label_dict.items():
        if _id == search_id:
            found_name = name
    if found_name is None:
        print(search_id, "not found!")
    assert found_name is not None
    return found_name


