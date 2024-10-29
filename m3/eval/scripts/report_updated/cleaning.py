#!/usr/bin/env python

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

import os
import re
import sys


def remove_newline(text):
    return text.replace("\n", "").replace("\r", "").strip()


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(paragraph):
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
    # Filter out sentences that contain the underscore character
    return [sentence for sentence in sentences if "_" not in sentence]


def refine_numbered_sentences(sentences):
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
    refined_sentences = []

    for sentence in sentences:
        # Check if the first character is a letter
        if sentence and sentence[0].isalpha():
            # Capitalize the first letter
            refined_sentences.append(sentence[0].upper() + sentence[1:])
        else:
            print(f"First character is not a letter in sentence: '{sentence}'")
            continue

    return refined_sentences


def remove_before_colon(sentences):
    # Replace only when a colon is followed by a space
    refined_sentences = [sentence.split(": ", 1)[-1] if ": " in sentence else sentence for sentence in sentences]

    return refined_sentences


def replace_abbreviations(text):
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
    # Split the report based on double newlines (i.e., paragraphs separated by empty lines)
    paragraphs = report.split("\n\n")

    # Remove leading/trailing spaces from each paragraph and exclude empty paragraphs
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

    return paragraphs


def remove_duplicate_sentences(sentences):
    seen = set()  # To keep track of sentences we have already seen
    unique_sentences = []

    for sentence in sentences:
        if sentence not in seen:
            unique_sentences.append(sentence)  # Add sentence to the result if not seen before
            seen.add(sentence)  # Mark this sentence as seen

    return unique_sentences


def remove_single_char_sentences(sentence_list):
    """
    This function removes sentences with only one character from the list.
    """
    return [sentence for sentence in sentence_list if len(sentence) > 6]


def find_duplicates(input_list):
    """
    This function returns a list of duplicated values in the input list.
    """
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)


def replace_appearance(text, target_word, replacement_word, nth_app):
    """
    Replace the second appearance of a target_word in the text with a replacement_word.
    """
    # Split the text at each occurrence of the target_word
    parts = text.split(target_word)

    if len(parts) <= 2:
        # If the target_word appears less than 2 times, no second replacement is needed
        return text

    # Rebuild the string: join the parts, but replace the second occurrence
    modified_text = target_word.join(parts[:nth_app]) + replacement_word + target_word.join(parts[nth_app:])

    return modified_text


def remove_question_sentences(sentences):
    """
    Remove sentences that contain question marks from the list.
    """
    return [sentence for sentence in sentences if "?" not in sentence]


def process_folders(predicted_folder, output_folder):
    """
    Processes the predicted .txt files by applying text transformations like lemmatization
    and sentence reordering based on semantic similarity. The processed files are saved
    in the output folder.

    Args:
        predicted_folder: Path to the folder containing predicted .txt files.
        output_folder: Path to save the processed .txt files.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through predicted files in the predicted folder
    for predicted_filename in os.listdir(predicted_folder):
        if predicted_filename.endswith(".txt"):
            predicted_file_path = os.path.join(predicted_folder, predicted_filename)

            # Read the predicted text
            # with open(predicted_file_path, "r") as pred_file:
            #     predicted_text = pred_file.read()
            with open(predicted_file_path) as f:
                report = "".join(f.readlines())

            report = remove_newline(report)
            report = normalize_spaces(report)
            # report = replace_abbreviations(report)

            sentences = split_into_sentences(report)
            sentences = remove_before_colon(sentences)
            # sentences = remove_sentences_with_underscore(sentences)
            # sentences = refine_numbered_sentences(sentences)
            sentences = skip_to_first_letter(sentences)
            sentences = remove_single_char_sentences(sentences)
            sentences = capitalize_first_letter(sentences)
            sentences = add_period_if_missing(sentences)
            sentences = remove_duplicate_sentences(sentences)
            sentences = remove_question_sentences(sentences)

            if len(sentences) < 1:
                continue

            out_report = " ".join(sentences)

            print(report)
            print(out_report)

            # Save the processed text to the output folder
            output_file_path = os.path.join(output_folder, predicted_filename)
            with open(output_file_path, "w") as output_file:
                output_file.write(out_report)

            print(f"Processed and saved: {predicted_filename}")


predicted_folder = sys.argv[1]
output_folder = sys.argv[2]

# Call the function to process the folders
process_folders(predicted_folder, output_folder)
