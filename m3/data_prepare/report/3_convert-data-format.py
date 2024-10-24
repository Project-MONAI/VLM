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
import json
import logging
import os
import pickle

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


def process_data(image_dir, text_dir, output_dir, annotation_file):
    """
    Process the images and corresponding reports and save them in a dataset.

    Args:
        image_dir (str): Directory containing the images.
        text_dir (str): Directory containing the reference reports.
        output_dir (str): Directory to save the processed dataset.
        annotation_file (str): Path to the mimic_annotation.json file.
    """
    # Load the JSON annotation file
    try:
        with open(annotation_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Annotation file not found: {annotation_file}")
        return
    except Exception as e:
        logger.error(f"Error reading annotation file {annotation_file}: {e}")
        return

    # Only process the "train" section of the data
    train_data = data.get("train", [])
    num_cases = len(train_data)
    data_dict = []

    for _i, item in enumerate(train_data):
        logger.info(f"Processing {_i + 1}/{num_cases}: {item['id']}")

        report = item.get("report", "")
        image_paths = item.get("image_path", [])

        # Process each image in the "image_path" list
        for image_path in image_paths:
            # Construct the full image path
            full_image_path = os.path.join(image_dir, image_path)

            # Encode image to base64
            image_base64_str = encode_image_to_base64(full_image_path)
            if image_base64_str is None:
                logger.warning(f"Failed to encode image, skipping: {full_image_path}")
                continue

            # Construct the corresponding text file path (replacing ".jpg" with ".txt")
            text_filename = os.path.basename(image_path).replace(".jpg", ".txt")
            full_text_path = os.path.join(text_dir, text_filename)

            # Read the corresponding text report
            try:
                with open(full_text_path, "r") as file:
                    reference_report = file.read()
            except FileNotFoundError:
                logger.error(f"Text file not found: {full_text_path}")
                continue
            except Exception as e:
                logger.error(f"Error reading text file {full_text_path}: {e}")
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
    IMAGE_DIR = "/path/to/images"  # Image directory
    TEXT_DIR = "./report_new"  # Text directory
    OUTPUT_DIR = "./gt"  # Output directory
    INPUT_JSON_FILENAME = "mimic_annotation.json"  # Path to mimic_annotation.json

    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Run the main processing function
    process_data(IMAGE_DIR, TEXT_DIR, OUTPUT_DIR, INPUT_JSON_FILENAME)
