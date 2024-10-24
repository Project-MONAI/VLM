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
import logging
import json
import openai
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the OpenAI API key is set as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("The OpenAI API key is not set. Please set it as an environment variable 'OPENAI_API_KEY'.")

# Constants
MODEL_NAME = "meta/llama-3.1-8b-instruct"  # or "meta/llama-3.1-70b-instruct"
INPUT_JSON_FILENAME = "mimic_annotation.json"
OUTPUT_DIR = "./report_new"
TEMPLATES_FILENAME = "sentence-pool.txt"


def load_json_data(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The parsed JSON content.

    Raises:
        FileNotFoundError: If the file at `file_path` is not found.
        Exception: For any other error encountered while reading the file.
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading or parsing JSON file {file_path}: {e}")
        raise


def load_templates(file_path):
    """
    Load template content from a file.

    Args:
        file_path (str): The path to the file containing template sentences.

    Returns:
        str: The content of the file as a string.

    Raises:
        FileNotFoundError: If the template file is not found at the specified path.
        Exception: For other errors encountered while reading the file.
    """
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Template file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading template file: {e}")
        raise


def initialize_output_directory(output_dir):
    """
    Ensure that the output directory exists, creating it if necessary.

    Args:
        output_dir (str): The path to the output directory to be created.

    Raises:
        Exception: If there is an error creating the directory.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory created: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        raise


def process_files(data, templates, output_dir):
    """
    Process all files found under the "train" and "test" keys in the JSON file,
    retrieve the reports, generate new reports using OpenAI API, and save the results.

    Args:
        data (dict): The loaded JSON data.
        templates (str): The template content to be used for generating reports.
        output_dir (str): The directory where the processed files will be saved.

    Raises:
        Exception: If there are issues calling the OpenAI API or saving files.
    """
    # Process both "train" and "test" sets
    for split in ["train", "test"]:
        for item in data.get(split, []):
            report = item.get("report", "")
            image_paths = item.get("image_path", [])

            # Generate new report
            new_report = generate_new_report(templates, report)

            # Save each report with a new filename derived from the image_path
            for image_path in image_paths:
                # Extract the basename of the .jpg file and replace .jpg with .txt
                filename = os.path.basename(image_path).replace(".jpg", ".txt")
                output_path = os.path.join(output_dir, filename)
                save_report(new_report, output_path)


def generate_new_report(templates, report):
    """
    Call OpenAI API to generate a new report based on the template and input report.

    Args:
        templates (str): The template content used for generating the new report.
        report (str): The input report that needs to be processed.

    Returns:
        str: The generated report after processing with the OpenAI API.

    Raises:
        Exception: If the OpenAI API call fails or an unexpected error occurs.
    """
    messages = [
        {
            "role": "system",
            "content": "You are an expert radiologist.",
        },
        {
            "role": "user",
            "content": f"{templates}\n\nPlease replace sentences with similar meanings in the contents below with the exact sentences from the template provided, "
            f"ensuring no other parts of the content are altered. Please directly output the updated report in the format 'new report: ...'.\n\n{report}",
        },
    ]

    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=2048,  # Adjust max tokens based on the expected response length.
            temperature=0.2,  # Set temperature for more deterministic results.
        )
        new_report = response["choices"][0]["message"]["content"]
        return new_report.replace("new report:", "").replace("New report:", "").strip()
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


def save_report(new_report, output_path):
    """
    Save the generated report to the output file.

    Args:
        new_report (str): The generated report content to be saved.
        output_path (str): The file path where the report will be saved.

    Raises:
        Exception: If the report cannot be saved to the specified path.
    """
    try:
        with open(output_path, "w") as f:
            f.write(new_report)
        logger.info(f"New report saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save report to {output_path}: {e}")
        raise


def main():
    """
    Main function to load JSON data, templates, and process the files.

    This function orchestrates the steps of loading the JSON content, processing each item
    in both "train" and "test" sets, generating new reports using the OpenAI API, and saving the
    results in the output directory.
    """
    try:
        # Load JSON data
        data = load_json_data(INPUT_JSON_FILENAME)

        # Load templates
        templates = load_templates(TEMPLATES_FILENAME)

        # Create output directory
        initialize_output_directory(OUTPUT_DIR)

        # Process files
        process_files(data, templates, OUTPUT_DIR)

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
