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

import copy
import json
import multiprocessing
import os
import random
import re
import sys
import warnings
from pathlib import Path

import jieba
import numpy as np
import section_parser as sp
import spacy

random.seed(10)


def remove_newline(text):
    """Remove newline characters from the text."""
    return text.replace("\n", "").replace("\r", "").strip()


def normalize_spaces(text):
    """Normalize spaces in the text."""
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(paragraph):
    """Split the paragraph into sentences."""
    # Regular expression to split sentences but avoid splitting on "a.m.", "p.m.", and "Dr."
    # The period following a.m., p.m., and Dr. is not treated as sentence-ending punctuation
    sentence_endings = re.compile(r"(?<!a\.m)(?<!p\.m)(?<!Dr)\.(?!\d)|(?<=[!?])")

    # Split based on the regex
    sentences = sentence_endings.split(paragraph)

    # Add the punctuation back to the sentences
    refined_sentences = []
    for i in range(len(sentences) - 1):
        sentence = sentences[i].strip()
        if sentence:
            # Add the appropriate punctuation back
            refined_sentences.append(
                sentence
                + paragraph[
                    paragraph.find(sentences[i])
                    + len(sentences[i]) : paragraph.find(sentences[i])
                    + len(sentences[i])
                    + 1
                ]
            )

    # Handle the last sentence if it was split correctly
    if sentences[-1].strip():
        refined_sentences.append(sentences[-1].strip())

    return refined_sentences


def remove_sentences_with_underscore(sentences):
    """Remove sentences that contain the underscore character."""
    # Filter out sentences that contain the underscore character
    return [sentence for sentence in sentences if "_" not in sentence]


def refine_numbered_sentences(sentences):
    """Refine sentences that start with a number followed by a period."""
    # Regular expression to match "1. ", "2. ", etc.
    pattern = re.compile(r"^\d+\.\s")

    refined_sentences = []
    for sentence in sentences:
        # Check if the sentence is just "1." or "2." etc.
        if re.fullmatch(r"\d+\.", sentence.strip()):
            continue  # Skip sentences that only contain "1." or "2." etc.
        # Refine the sentence by removing the number followed by ". "
        refined_sentences.append(pattern.sub("", sentence))

    return refined_sentences


def skip_to_first_letter(sentences):
    """Skip to the first letter in each sentence."""
    refined_sentences = []

    for sentence in sentences:
        # Find the position of the first letter
        for i, char in enumerate(sentence):
            if char.isalpha():
                # Capitalize the first letter and append the refined sentence
                refined_sentences.append(sentence[i].upper() + sentence[i + 1 :])
                break
        else:
            # If no letter is found in the sentence, keep it as is
            refined_sentences.append(sentence)

    return refined_sentences


def capitalize_first_letter(sentences):
    """Capitalize the first letter of each sentence."""
    refined_sentences = []

    for sentence in sentences:
        # Check if the first character is a letter
        if sentence and sentence[0].isalpha():
            # Capitalize the first letter
            refined_sentences.append(sentence[0].upper() + sentence[1:])
        else:
            # Raise an error if the first character is not a letter
            raise ValueError(f"First character is not a letter in sentence: '{sentence}'")

    return refined_sentences


def remove_before_colon(sentences):
    """Remove text before a colon in each sentence."""
    # Replace only when a colon is followed by a space
    refined_sentences = [sentence.split(": ", 1)[-1] if ": " in sentence else sentence for sentence in sentences]

    return refined_sentences


def replace_abbreviations(text):
    """Replace common abbreviations in the text."""
    # Replace specific abbreviations
    pattern = re.compile(r"a\.m\.\s([A-Z])")
    text = pattern.sub(r"am_. \1", text)
    pattern = re.compile(r"p\.m\.\s([A-Z])")
    text = pattern.sub(r"pm_. \1", text)

    text = text.replace("a.m.", "am_")
    text = text.replace("p.m.", "pm_")
    text = text.replace("Dr.", "Dr_")
    text = text.replace(" VIEW ", " VIEW_. ")
    text = text.replace("_: ", "_. ")

    return text


def add_period_if_missing(sentences):
    """Add a period at the end of each sentence if missing."""
    # Regular expression to check if the sentence ends with a punctuation mark
    pattern = re.compile(r"[.!?]$")

    refined_sentences = []
    for sentence in sentences:
        # If the sentence doesn't end with punctuation, add a period
        if not pattern.search(sentence.strip()):
            refined_sentences.append(sentence.strip() + ".")
        else:
            refined_sentences.append(sentence.strip())

    return refined_sentences


def partition_into_paragraphs(report):
    """Partition the report into paragraphs."""
    # Split the report based on double newlines (i.e., paragraphs separated by empty lines)
    paragraphs = report.split("\n\n")

    # Remove leading/trailing spaces from each paragraph and exclude empty paragraphs
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

    return paragraphs


def remove_duplicate_sentences(sentences):
    """Remove duplicate sentences from the list."""
    seen = set()  # To keep track of sentences we have already seen
    unique_sentences = []

    for sentence in sentences:
        if sentence not in seen:
            unique_sentences.append(sentence)  # Add sentence to the result if not seen before
            seen.add(sentence)  # Mark this sentence as seen

    return unique_sentences


def remove_single_char_sentences(sentence_list):
    """This function removes sentences with only one character from the list."""
    return [sentence for sentence in sentence_list if len(sentence) > 6]


def find_duplicates(input_list):
    """This function returns a list of duplicated values in the input list."""
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)


def replace_appearance(text, target_word, replacement_word, nth_app):
    """Replace the second appearance of a target_word in the text with a replacement_word."""
    # Split the text at each occurrence of the target_word
    parts = text.split(target_word)

    if len(parts) <= 2:
        # If the target_word appears less than 2 times, no second replacement is needed
        return text

    # Rebuild the string: join the parts, but replace the second occurrence
    modified_text = target_word.join(parts[:nth_app]) + replacement_word + target_word.join(parts[nth_app:])

    return modified_text


def remove_question_sentences(sentences):
    """Remove sentences that contain question marks from the list."""
    return [sentence for sentence in sentences if "?" not in sentence]


def get_mimic_data(text_filepath):
    """Extract the relevant text from the MIMIC CXR report."""
    path_to_report = text_filepath
    if not os.path.exists(path_to_report):
        print(f"path_to_report: {path_to_report}")
        input()
        return None

    with open(path_to_report) as f:
        report = "".join(f.readlines())

    sections, section_names, _ = sp.section_text(report)

    out_report = ""
    for _key in section_names:
        if _key in [
            "finding",
            "findings",
            "impression",
            "impressions",
            "last_paragraph",
            "report",
        ]:
            _index = section_names.index(_key)
            _part = sections[_index]
            out_report += _part

    out_report = remove_newline(out_report)
    out_report = normalize_spaces(out_report)
    out_report = replace_abbreviations(out_report)

    sentences = split_into_sentences(out_report)
    sentences = remove_before_colon(sentences)
    sentences = remove_sentences_with_underscore(sentences)
    sentences = refine_numbered_sentences(sentences)
    sentences = skip_to_first_letter(sentences)
    sentences = remove_single_char_sentences(sentences)
    sentences = capitalize_first_letter(sentences)
    sentences = add_period_if_missing(sentences)
    sentences = remove_duplicate_sentences(sentences)
    sentences = remove_question_sentences(sentences)

    if len(sentences) > 0:
        return " ".join(sentences)

    return None


list_filepath = "./list.txt"
output_dir = "./text_gt/training"
report_dir = "./reports"

filepaths = [_item for _item in filepaths if _item.endswith(".txt")]

num_cases = len(filepaths)

for _i in range(num_cases):
    os.system("clear")

    print(f"{_i + 1} / {num_cases}\n")
    text_filepath = filepaths[_i]

    new_text = get_mimic_data(os.path.join(report_dir, text_filepath))
    if new_text == None:
        continue

    output_filepath = os.path.join(output_dir, image_filename + ".txt")
    with open(output_filepath, "w") as output_file:
        output_file.write(new_text)
