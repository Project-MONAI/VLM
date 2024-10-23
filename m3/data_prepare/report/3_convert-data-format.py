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

import base64
import os
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path):
    """
    Encode a 2D image to a base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file does not exist: {image_path}")
        return None

    try:
        with open(image_path, "rb") as img_file:
            base64_string = base64.b64encode(img_file.read()).decode("utf-8")
        return base64_string
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None


def save_dataset(dataset_type, dataset_name, save_path, data):
    """
    Save the dataset to a pickle file.

    Args:
        dataset_type (str): Type of the dataset (e.g., 'captioning').
        dataset_name (str): Name of the dataset (e.g., 'mimic_train').
        save_path (str): Directory to save the dataset.
        data (list): The dataset to be saved.

    Raises:
        Exception: If saving the dataset fails.
    """
    save_filename = f"{dataset_type}_{dataset_name}.pkl"
    save_pathname = os.path.join(save_path, save_filename)

    try:
        with open(save_pathname, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Dataset saved successfully at {save_pathname}")
    except Exception as e:
        logger.error(f"Error saving dataset to {save_pathname}: {e}")
        raise


def load_file_paths(list_filepath):
    """
    Load file paths from the provided list file.

    Args:
        list_filepath (str): Path to the list file containing file names.

    Returns:
        list: List of file paths.
    """
    try:
        with open(list_filepath, "r") as file:
            filepaths = file.readlines()
        return [path.strip() for path in filepaths]
    except FileNotFoundError:
        logger.error(f"List file not found: {list_filepath}")
        raise
    except Exception as e:
        logger.error(f"Error reading list file {list_filepath}: {e}")
        raise


def main(image_dir, text_dir, output_dir, list_filepath):
    """
    Main function to process images and corresponding reports and save them in a dataset.

    Args:
        image_dir (str): Directory containing the images.
        text_dir (str): Directory containing the reference reports.
        output_dir (str): Directory to save the processed dataset.
        list_filepath (str): Path to the list file containing image and report filenames.
    """
    # Load the file paths
    try:
        filepaths = load_file_paths(list_filepath)
    except Exception as e:
        logger.error(f"Failed to load file paths: {e}")
        return

    num_cases = len(filepaths)
    data_dict = []

    for _i, file_name in enumerate(filepaths):
        logger.info(f"Processing {_i + 1}/{num_cases}: {file_name}")

        image_filepath = os.path.join(image_dir, file_name.replace(".txt", ""))
        text_filepath = os.path.join(text_dir, file_name)

        # Check if the image file exists
        if not os.path.exists(image_filepath):
            logger.warning(f"Image file not found, skipping: {image_filepath}")
            continue

        # Encode image to base64
        image_base64_str = encode_image_to_base64(image_filepath)
        if image_base64_str is None:
            logger.warning(f"Failed to encode image, skipping: {image_filepath}")
            continue

        # Read the corresponding text report
        try:
            with open(text_filepath, "r") as file:
                reference_report = file.read()
        except FileNotFoundError:
            logger.error(f"Text file not found: {text_filepath}")
            continue
        except Exception as e:
            logger.error(f"Error reading text file {text_filepath}: {e}")
            continue

        # Create a data entry
        data_entry = {
            "question": "Describe the image in detail.",
            "image": [image_base64_str],
            "answer": reference_report,
        }
        data_dict.append(data_entry)

    # Save the processed dataset
    try:
        save_dataset("captioning", "mimic_train", output_dir, data_dict)
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")


if __name__ == "__main__":
    IMAGE_DIR = "./images"
    TEXT_DIR = "./text_gt"
    OUTPUT_DIR = "./gt"
    LIST_FILEPATH = "./list.txt"

    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Run the main processing function
    main(IMAGE_DIR, TEXT_DIR, OUTPUT_DIR, LIST_FILEPATH)
