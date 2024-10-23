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
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API client
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure the OpenAI API key is set as an environment variable
if not openai.api_key:
    raise ValueError("The OpenAI API key is not set. Please set it as an environment variable 'OPENAI_API_KEY'.")

# Constants
MODEL_NAME = "meta/llama-3.1-8b-instruct"
TEXT_FILENAME = "./sentences_test_slim.txt"
TEMPLATES_FILENAME = "./templates_sentences_test_slim.txt"
BATCH_SIZE = 100


def load_file_lines(file_path):
    """
    Load the lines from a file, stripping newline characters.

    Args:
        file_path (str): Path to the file to load.

    Returns:
        list: A list of lines, each stripped of newline characters.

    Raises:
        FileNotFoundError: If the file at `file_path` is not found.
        Exception: For any other error encountered while reading the file.
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
        return [line.strip() for line in lines]
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise


def make_api_call(sentences, templates):
    """
    Make a call to the OpenAI API using the provided sentences and templates.

    Args:
        sentences (list): A list of sentences to process.
        templates (str): A string of template sentences to use for replacement.

    Returns:
        str: The processed text returned by the OpenAI API.

    Raises:
        Exception: If the OpenAI API call fails for any reason.
    """
    combined_sentences = "\n".join(sentences) + f"\n{templates}"

    messages = [
        {
            "role": "system",
            "content": "You are an expert radiologist.",
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
            Here is the list of sentences to process: {combined_sentences}. \
            Please provide the final list of simplified sentences, focusing only on the most common meanings.",
        },
    ]

    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=2048,  # Adjust the token limit based on expected response length
            temperature=0.2,  # Set temperature for more deterministic results
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


def save_template(new_template, output_path):
    """
    Save the generated template to a specified file.

    Args:
        new_template (str): The content to be saved as the template.
        output_path (str): Path where the new template will be saved.

    Raises:
        Exception: If there is any issue saving the template to the file.
    """
    try:
        with open(output_path, "w") as f:
            f.write(new_template)
        logger.info(f"New template saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save template to {output_path}: {e}")
        raise


def main():
    """
    Main function to load data, call the OpenAI API, and save the processed templates.

    The function loads sentences from a file, splits them into chunks, processes them with the OpenAI API,
    and then saves the processed templates back to a file. Templates are loaded if present, and API results
    are logged during the process.
    """
    try:
        lines = load_file_lines(TEXT_FILENAME)

        # Load templates if the file exists
        templates = ""
        if os.path.exists(TEMPLATES_FILENAME):
            templates = load_file_lines(TEMPLATES_FILENAME)

        for i in range(0, len(lines), BATCH_SIZE):
            chunk = lines[i : i + BATCH_SIZE]
            logger.info(f"Processing chunk {i // BATCH_SIZE + 1} of {len(lines) // BATCH_SIZE + 1}")

            new_template = make_api_call(chunk, templates)
            save_template(new_template, TEMPLATES_FILENAME)

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()
