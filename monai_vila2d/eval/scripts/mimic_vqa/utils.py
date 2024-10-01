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


from pathlib import Path
import json
from typing import Any, List, Tuple, Union, Optional
from collections import defaultdict
import math


def load_json(path: Path) -> Any:
    """
    Loads a JSON file from the specified path.

    Args:
        path: A Path representing the file path.

    Returns:
        The data loaded from the JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path: Path, data: Any) -> None:
    """
    Saves data to a JSON file at the specified path.

    Args:
        path: A Path representing the file path.
        data: The data to be serialized and saved.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_jsonl(path: Path) -> List[Any]:
    """
    Loads a JSONL file (JSON lines) from the specified path.

    Args:
        path: A Path representing the file path.

    Returns:
        A list of data loaded from the JSONL file.
    """
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def save_jsonl(path: Path, data: List[Any]) -> None:
    """
    Saves data to a JSONL file at the specified path.

    Args:
        path: A Path representing the file path.
        data: A list of data to be serialized and saved.
    """
    with open(path, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')

def maybe_make_dir(path: str) -> Path:
    """
    Creates a directory at the specified path if it does not exist.

    Args:
        path: A string representing the directory path.

    Returns:
        A Path object for the created or existing directory.
    """
    dir = Path(path)
    if not dir.exists():
        dir.mkdir(parents=True)
    return dir


def brevity_penalty(candidate, references):
    c = len(candidate)
    ref_lens = (len(reference) for reference in references)
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))
    
    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)

def modified_precision(candidate, references, n):
    max_frequency = defaultdict(int)
    min_frequency = defaultdict(int)
    
    candidate_words = split_sentence(candidate, n)
    
    for reference in references:
        reference_words = split_sentence(reference, n)
        for word in candidate_words:
            max_frequency[word] = max(max_frequency[word], reference_words[word])
    for word in candidate_words:
            min_frequency[word] = min(max_frequency[word], candidate_words[word])
    P = sum(min_frequency.values()) / sum(candidate_words.values())
    return P

def split_sentence(sentence, n):
    words = defaultdict(int)
    # tmp_sentence = re.sub("[^a-zA-Z ]", "", sentence)
    tmp_sentence = sentence
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words
