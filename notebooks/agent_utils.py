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
import itertools
import logging
import os
import re
import sys
import tempfile
import time
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from shutil import copyfile, move, rmtree
from uuid import uuid4

# Third-party imports
import nibabel as nib
import numpy as np
import requests
import skimage
import torch
from monai.bundle import create_workflow
from monai.transforms import (
    Compose,
    LoadImageD,
    MapTransform,
    OrientationD,
    ScaleIntensityD,
    ScaleIntensityRangeD,
)
from PIL import Image
from tqdm import tqdm

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Constants
SYS_PROMPT = None  # only useful in local mode
REMOTE_URL = "https://developer.download.nvidia.com/assets/Clara/monai/samples"
SEGMENTATION_TOKEN = "<image>"

MODEL_CARDS = (
    "Here is a list of available expert models:\n"
    "<BRATS(args)> "
        "Modality: MRI, "
        "Task: segmentation, "
        "Overview: A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data, "
        "Accuracy: Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 - Average: 0.8518, "
        "Valid args are: None\n"
    "<VISTA3D(args)> "
        "Modality: CT, "
        "Task: segmentation, "
        "Overview: domain-specialized interactive foundation model developed for segmenting and annotating human anatomies with precision, "
        "Accuracy: 127 organs: 0.792 Dice on average, "
        "Valid args are: 'everything', 'hepatic tumor', 'pancreatic tumor', 'lung tumor', 'bone lesion', 'organs', 'cardiovascular', 'gastrointestinal', 'skeleton', or 'muscles'\n"
    "<VISTA2D(args)> "
        "Modality: cell imaging, "
        "Task: segmentation, "
        "Overview: model for cell segmentation, which was trained on a variety of cell imaging outputs, including brightfield, phase-contrast, fluorescence, confocal, or electron microscopy, "
        "Accuracy: Good accuracy across several cell imaging datasets, "
        "Valid args are: None\n"
    "<CXR(args)> "
        "Modality: chest x-ray (CXR), "
        "Task: classification, "
        "Overview: pre-trained model which are trained on large cohorts of data, "
        "Accuracy: Good accuracy across several diverse chest x-rays datasets, "
        "Valid args are: None\n"
    "Give the model <NAME(args)> when selecting a suitable expert model.\n"
)


MODALITY_MAP = {
    "cxr": "CXR",
    "chest x-ray": "CXR",
    "ct image": "CT",
    "mri": "MRI",
    "magnetic resonance imaging": "MRI",
    "ultrasound": "US",
    "cell imaging": "cell imaging",
}


VISTA_LABEL_DICT = {
    "everything": "../experts/vista3d/label_dict.json",
    "hepatic tumor": {
        "liver": 1,
        "hepatic tumor": 26
    },
    "hepatoma": {
        "liver": 1,
        "hepatic tumor": 26
    },
    "pancreatic tumor": {
        "pancreas": 4,
        "pancreatic tumor": 24
    },
    "lung tumor": {
        "lung": 20,
        "lung tumor": 23,
        "left lung upper lobe": 28,
        "left lung lower lobe": 29,
        "right lung upper lobe": 30,
        "right lung middle lobe": 31,
        "right lung lower lobe": 32
    },
    "bone lesion": {
        "bone lesion": 128
    },
    "organs": {
        "liver": 1,
        "kidney": 2,
        "spleen": 3,
        "pancreas": 4,
        "right kidney": 5,
        "right adrenal gland": 8,
        "left adrenal gland": 9,
        "gallbladder": 10,
        "left kidney": 14,
        "brain": 22,
        "lung tumor": 23,
        "pancreatic tumor": 24,
        "hepatic vessel": 25,
        "hepatic tumor": 26,
        "colon cancer primaries": 27,
        "left lung upper lobe": 28,
        "left lung lower lobe": 29,
        "right lung upper lobe": 30,
        "right lung middle lobe": 31,
        "right lung lower lobe": 32,
        "trachea": 57,
        "left kidney cyst": 116,
        "right kidney cyst": 117,
        "prostate": 118,
        "spinal cord": 121,
        "thyroid gland": 126,
        "airway": 132
    },
    "cardiovascular": {
        "aorta": 6,
        "inferior vena cava": 7,
        "portal vein and splenic vein": 17,
        "left iliac artery": 58,
        "right iliac artery": 59,
        "left iliac vena": 60,
        "right iliac vena": 61,
        "left atrial appendage": 108,
        "brachiocephalic trunk": 109,
        "left brachiocephalic vein": 110,
        "right brachiocephalic vein": 111,
        "left common carotid artery": 112,
        "right common carotid artery": 113,
        "heart": 115,
        "pulmonary vein": 119,
        "left subclavian artery": 123,
        "right subclavian artery": 124,
        "superior vena cava": 125
    },
    "gastrointestinal": {
        "esophagus": 11,
        "stomach": 12,
        "duodenum": 13,
        "bladder": 15,
        "small bowel": 19,
        "colon": 62
    },
    "skeleton": {
        "bone": 21,
        "vertebrae L5": 33,
        "vertebrae L4": 34,
        "vertebrae L3": 35,
        "vertebrae L2": 36,
        "vertebrae L1": 37,
        "vertebrae T12": 38,
        "vertebrae T11": 39,
        "vertebrae T10": 40,
        "vertebrae T9": 41,
        "vertebrae T8": 42,
        "vertebrae T7": 43,
        "vertebrae T6": 44,
        "vertebrae T5": 45,
        "vertebrae T4": 46,
        "vertebrae T3": 47,
        "vertebrae T2": 48,
        "vertebrae T1": 49,
        "vertebrae C7": 50,
        "vertebrae C6": 51,
        "vertebrae C5": 52,
        "vertebrae C4": 53,
        "vertebrae C3": 54,
        "vertebrae C2": 55,
        "vertebrae C1": 56,
        "skull": 120,
        "sternum": 122,
        "vertebrae S1": 127,
        "bone lesion": 128,
        "left rib 1": 63,
        "left rib 2": 64,
        "left rib 3": 65,
        "left rib 4": 66,
        "left rib 5": 67,
        "left rib 6": 68,
        "left rib 7": 69,
        "left rib 8": 70,
        "left rib 9": 71,
        "left rib 10": 72,
        "left rib 11": 73,
        "left rib 12": 74,
        "right rib 1": 75,
        "right rib 2": 76,
        "right rib 3": 77,
        "right rib 4": 78,
        "right rib 5": 79,
        "right rib 6": 80,
        "right rib 7": 81,
        "right rib 8": 82,
        "right rib 9": 83,
        "right rib 10": 84,
        "right rib 11": 85,
        "right rib 12": 86,
        "left humerus": 87,
        "right humerus": 88,
        "left scapula": 89,
        "right scapula": 90,
        "left clavicula": 91,
        "right clavicula": 92,
        "left femur": 93,
        "right femur": 94,
        "left hip": 95,
        "right hip": 96,
        "sacrum": 97,
        "costal cartilages": 114
    },
    "muscles": {
        "left gluteus maximus": 98,
        "right gluteus maximus": 99,
        "left gluteus medius": 100,
        "right gluteus medius": 101,
        "left gluteus minimus": 102,
        "right gluteus minimus": 103,
        "left autochthon": 104,
        "right autochthon": 105,
        "left iliopsoas": 106,
        "right iliopsoas": 107
    }
}

class Dye(MapTransform):
    """
    Dye the label map with predefined colors and write the image and label to disk.

    Args:
        slice_index: the index of the slice to be dyed. If None, the middle slice will be picked.
        axis: the axis of the slice.
        image_key: the key to extract the image data.
        label_key: the key to extract the label data.
        image_filename: the filename to save the image.
        label_filename: the filename to save the label.
        output_dir: the directory to save the image and label.
        bg_label: the label value for the background.
    """

    COLORS = [
        "red",
        "blue",
        "yellow",
        "magenta",
        "green",
        "indigo",
        "darkorange",
        "cyan",
        "pink",
        "brown",
        "orange",
        "lime",
        "orange",
        "gold",
        "yellowgreen",
        "darkgreen",
    ]

    def __init__(
        self,
        slice_index: int | None = None,
        axis: int = 2,
        image_key: str = "image",
        label_key: str = "label",
        image_filename: str = "image.jpg",
        label_filename: str = "label.jpg",
        output_dir: Path = Path("."),
        bg_label: int = 0,
    ):
        """Initialize the dye transform."""
        self.slice_index = slice_index
        self.axis = axis
        self.image_key = image_key
        self.label_key = label_key
        self.image_filename = image_filename
        self.label_filename = label_filename
        self.output_dir = Path(output_dir)
        self.bg_label = bg_label
        self.keys = [self.image_key, self.label_key]
        self.allow_missing_keys = True

    def __call__(self, data):
        """Dye the label map with predefined colors and write the image and label to disk."""
        d = dict(data)
        for key in self.key_iterator(d):
            np_array = np.squeeze(d.get(key))
            slice_index = np_array.shape[2] // 2 if self.slice_index is None else self.slice_index
            slice = np.take(np_array, slice_index, axis=self.axis)
            d[key] = np.rot90(np.swapaxes(slice.astype(np.uint8), 0, 1), k=2)

        os.makedirs(self.output_dir, exist_ok=True)
        skimage.io.imsave(self.output_dir / self.image_filename, np.stack([d[self.image_key]] * 3, axis=-1))

        if self.label_key in d:
            color_label = (
                skimage.color.label2rgb(
                    d[self.label_key], colors=self.COLORS, image=d[self.image_key], bg_label=self.bg_label
                )
                * 255
            )

            skimage.io.imsave(self.output_dir / self.label_filename, color_label.astype(np.uint8))

            unique_labels = np.unique(d[self.label_key])
            color_cyle = itertools.cycle(Dye.COLORS)

            colormap = {}
            unique_labels = unique_labels[unique_labels != self.bg_label]  # remove background label
            for label_id, label_color in zip(unique_labels, color_cyle):
                colormap[label_id] = label_color
            d["colormap"] = colormap
        return d


def load_image(image_path_or_data_url: str) -> Image:
    """
    Load the image from the URL.

    Args:
        image: the image URL or the base64 encoded image that starts with "data:image".

    Returns:
        PIL.Image: the loaded image.
    """
    logger.debug(f"Loading image from URL")

    if os.path.exists(image_path_or_data_url):
        try:
            return Image.open(image_path_or_data_url).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load the image: {e}")
    else:
        image_base64_regex = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")
        match_results = image_base64_regex.match(image_path_or_data_url)
        if match_results:
            image_base64 = match_results.groups()[1]
            return Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")

    raise ValueError(f"Unable to load the image from {image_path_or_data_url[:50]}")


def save_image_url_to_file(image_url: str, output_dir: Path) -> str:
    """Save the image from the URL to the output directory"""
    file_name = os.path.join(output_dir, image_url.split("/")[-1])
    # avoid re-downloading the image if it's already downloaded before
    if not os.path.exists(file_name):
        url_response = requests.get(image_url, allow_redirects=True)
        with open(file_name, "wb") as f:
            f.write(url_response.content)
    return file_name


def get_slice_filenames(image_file: str, slice_index: int, ext: str = "jpg"):
    """Small helper function to get the slice filenames"""
    base_name = os.path.basename(image_file)
    return base_name.replace(".nii.gz", f"_slice{slice_index}_img.{ext}")


def _ct_chat_template(input_text):
    """Apply the chat template to the input text"""
    return MODEL_CARDS + "<image>" + "This is a CT image.\n" + input_text


def apply_chat_template(input_text, with_image=True):
    """Apply the chat template to the input text"""
    if with_image:
        return _ct_chat_template(input_text)
    else:
        return input_text


def get_monai_transforms(
    keys,
    output_dir: Path | str,
    image_key="image",
    modality: str = "CT",
    slice_index: int | None = None,
    axis: int = 2,
    image_filename: str = "image.jpg",
    label_filename: str = "label.jpg",
):
    """
    Get the MONAI transforms for the modality.

    Args:
        keys: the keys.
        output_dir: the output directory.
        image_key: the image key.
        modality: the modality.
        slice_index: the slice index.
        axis: the axis.
    """
    logger.debug(f"Getting MONAI transforms for modality: {modality}")
    if image_key not in keys:
        raise ValueError(f"Image key {image_key} not found in the keys: {keys}")

    if modality == "CT":
        # abdomen soft tissue https://radiopaedia.org/articles/windowing-ct
        window_center = 50
        window_width = 400
        scaler = ScaleIntensityRangeD(
            keys=[image_key],
            a_min=window_center - window_width / 2,
            a_max=window_center + window_width / 2,
            b_min=0,
            b_max=255,
            clip=True,
        )
    elif modality == "MRI":
        scaler = ScaleIntensityD(keys=[image_key], minv=0, maxv=255, channel_wise=True)
    else:
        raise ValueError(f"Unsupported modality: {modality}. Supported modalities are 'CT' and 'MRI'.")

    return Compose(
        [
            LoadImageD(keys=keys, ensure_channel_first=True),
            OrientationD(keys=keys, axcodes="RAS"),
            scaler,
            Dye(
                slice_index=slice_index,
                axis=axis,
                output_dir=output_dir,
                image_filename=image_filename,
                label_filename=label_filename,
            ),
        ]
    )


def image_to_data_url(image, format="JPEG", max_size=None):
    """
    Convert an image to a data URL.

    Args:
        image (str | np.Array): The image to convert. If a string, it is treated as a file path.
        format (str): The format to save the image in. Default is "JPEG".
        max_size (tuple): The maximum size of the image. Default is None.
    """
    if isinstance(image, str) and os.path.exists(image):
        img = Image.open(image)
    else:
        raise ValueError(f"Invalid image type: {type(image)}")
    if max_size is not None:
        # Resize the image to the specified maximum height
        img.thumbnail(max_size)
    # Create a BytesIO buffer to save the image
    buffered = BytesIO()
    # Save the image to the buffer in the specified format
    img.save(buffered, format=format)
    # Convert the buffer content into bytes
    img_byte = buffered.getvalue()
    # Encode the bytes to base64
    img_base64 = base64.b64encode(img_byte).decode()
    # Convert the base64 bytes to string and format the data URL
    return f"data:image/{format.lower()};base64,{img_base64}"


def _get_modality_url(image_url_or_path: str | None):
    """
    Extract image modality by checking the URL or file path.
    If the URL or file path contains ".nii.gz" and contain "mri_", then it is MRI, else it is CT.
    If it contains "cxr_" then it is CXR, otherwise it is Unknown.
    """
    if isinstance(image_url_or_path, list) and len(image_url_or_path) > 0:
        image_url_or_path = image_url_or_path[0]
    if not isinstance(image_url_or_path, str):
        return "Unknown"
    if image_url_or_path.startswith("data:image"):
        return "Unknown"
    if ".nii.gz" in image_url_or_path.lower():
        if "mri_" in image_url_or_path.lower():
            return "MRI"
        return "CT"
    if "cxr_" in image_url_or_path.lower():
        return "CXR"
    return "Unknown"


def _get_modality_text(text: str):
    """Get the modality from the text"""
    
    if not text:
        return "Unknown"
    for keyword, modality in MODALITY_MAP.items():
        if keyword.lower() in text.lower():
            return modality
    return "Unknown"


def get_modality(image_url: str | None, text: str | None = None):
    """Get the modality from the image URL or text"""
    logger.debug(f"Getting modality from image URL or text")
    modality = _get_modality_url(image_url)
    if modality != "Unknown":
        return modality
    return _get_modality_text(text)


class ChatHistory:
    """Class to store the chat history"""

    def __init__(self):
        """
        Messages are stored as a list, with a sample format:

        messages = [
        # --------------- Below is the previous prompt from the user ---------------
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in the image? <image>"
                },
                {
                    "type": "image_path",
                    "image_path": image_path
                }
            ]
        },
        # --------------- Below is the answer from the previous completion ---------------
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": answer1,
                }
            ]
        },
        ]
        """
        self.messages = []
        self.last_prompt_with_image = None

    def append(self, prompt_or_answer, image_path=None, role="user"):
        """
        Append a new message to the chat history.

        Args:
            prompt_or_answer (str): The text prompt from human or answer from AI to append.
            image_url (str): The image file path to append.
            slice_index (int): The slice index for 3D images.
            role (str): The role of the message. Default is "user". Other option is "assistant" and "expert".
        """
        new_contents = [
            {
                "type": "text",
                "text": prompt_or_answer,
            }
        ]
        if image_path is not None:
            new_contents.append(
                {
                    "type": "image_path",
                    "image_path": image_path,
                }
            )
            self.last_prompt_with_image = prompt_or_answer

        self.messages.append({"role": role, "content": new_contents})


class ImageCache:
    """A simple image cache to store images and data URLs."""

    def __init__(self, cache_dir: Path):
        """Initialize the image cache."""
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        self.cache_dir = cache_dir
        self.cache_images = {}

    def cache(self, image_urls_or_paths):
        """Cache the images from the URLs or paths."""
        logger.debug(f"Caching the image to {self.cache_dir}")
        for _, items in image_urls_or_paths.items():
            items = items if isinstance(items, list) else [items]
            for item in items:
                if item.startswith("http"):
                    self.cache_images[item] = save_image_url_to_file(item, self.cache_dir)
                elif os.path.exists(item):
                    # move the file to the cache directory
                    file_name = os.path.basename(item)
                    self.cache_images[item] = os.path.join(self.cache_dir, file_name)
                    if not os.path.isfile(self.cache_images[item]):
                        copyfile(item, self.cache_images[item])

                if self.cache_images[item].endswith(".nii.gz"):
                    data = nib.load(self.cache_images[item]).get_fdata()
                    for slice_index in tqdm(range(data.shape[2])):
                        image_filename = get_slice_filenames(self.cache_images[item], slice_index)
                        if not os.path.exists(os.path.join(self.cache_dir, image_filename)):
                            compose = get_monai_transforms(
                                ["image"],
                                self.cache_dir,
                                modality=get_modality(item),
                                slice_index=slice_index,
                                image_filename=image_filename,
                            )
                            compose({"image": self.cache_images[item]})

    def cleanup(self):
        """Clean up the cache directory."""
        logger.debug(f"Cleaning up the cache")
        rmtree(self.cache_dir)

    def dir(self):
        """Return the cache directory."""
        return str(self.cache_dir)

    def get(self, key: str | list, default=None, list_return=False):
        """Get the image or data URL from the cache."""
        if isinstance(key, list):
            items = [self.cache_images.get(k) for k in key]
            return items if list_return else items[0]
        return self.cache_images.get(key, default)


class SessionVariables:
    """Class to store the session variables"""

    def __init__(self):
        """Initialize the session variables"""
        global SYS_PROMPT
        self.sys_prompt = SYS_PROMPT
        self.sys_msg = MODEL_CARDS
        self.use_model_cards = True
        self.slice_index = None  # Slice index for 3D images
        self.image_url = None  # Image URL to the image on the web
        self.backup = {}  # Cached varaiables from previous messages for the current conversation
        self.axis = 2
        self.top_p = 0.9
        self.temperature = 0.0
        self.max_tokens = 1024
        self.temp_working_dir = None
        # self.idx_range = (None, None)
        # self.interactive = False
        self.sys_msgs_to_hide = []
        self.modality_prompt = "Auto"

    def restore_from_backup(self, attr):
        """Retrieve the attribute from the backup"""
        attr_val = self.backup.get(attr, None)
        if attr_val is not None:
            self.__setattr__(attr, attr_val)


class ExpertVista3D():
    """Expert model for VISTA-3D."""

    def __init__(self) -> None:
        """Initialize the VISTA-3D expert model."""
        self.model_name = "VISTA3D"
        self.bundle_root = os.path.expanduser("~/.cache/torch/hub/bundle/vista3d_v0.5.4/vista3d")

    def _get_label_groups(self):
        """Get the label groups from the label groups path."""
        return {
            "everything": "../experts/vista3d/label_dict.json",
            "hepatic tumor": {
                "liver": 1,
                "hepatic tumor": 26
            },
            "hepatoma": {
                "liver": 1,
                "hepatic tumor": 26
            },
            "pancreatic tumor": {
                "pancreas": 4,
                "pancreatic tumor": 24
            },
            "lung tumor": {
                "lung": 20,
                "lung tumor": 23,
                "left lung upper lobe": 28,
                "left lung lower lobe": 29,
                "right lung upper lobe": 30,
                "right lung middle lobe": 31,
                "right lung lower lobe": 32
            },
            "bone lesion": {
                "bone lesion": 128
            },
            "organs": {
                "liver": 1,
                "kidney": 2,
                "spleen": 3,
                "pancreas": 4,
                "right kidney": 5,
                "right adrenal gland": 8,
                "left adrenal gland": 9,
                "gallbladder": 10,
                "left kidney": 14,
                "brain": 22,
                "lung tumor": 23,
                "pancreatic tumor": 24,
                "hepatic vessel": 25,
                "hepatic tumor": 26,
                "colon cancer primaries": 27,
                "left lung upper lobe": 28,
                "left lung lower lobe": 29,
                "right lung upper lobe": 30,
                "right lung middle lobe": 31,
                "right lung lower lobe": 32,
                "trachea": 57,
                "left kidney cyst": 116,
                "right kidney cyst": 117,
                "prostate": 118,
                "spinal cord": 121,
                "thyroid gland": 126,
                "airway": 132
            },
            "cardiovascular": {
                "aorta": 6,
                "inferior vena cava": 7,
                "portal vein and splenic vein": 17,
                "left iliac artery": 58,
                "right iliac artery": 59,
                "left iliac vena": 60,
                "right iliac vena": 61,
                "left atrial appendage": 108,
                "brachiocephalic trunk": 109,
                "left brachiocephalic vein": 110,
                "right brachiocephalic vein": 111,
                "left common carotid artery": 112,
                "right common carotid artery": 113,
                "heart": 115,
                "pulmonary vein": 119,
                "left subclavian artery": 123,
                "right subclavian artery": 124,
                "superior vena cava": 125
            },
            "gastrointestinal": {
                "esophagus": 11,
                "stomach": 12,
                "duodenum": 13,
                "bladder": 15,
                "small bowel": 19,
                "colon": 62
            },
            "skeleton": {
                "bone": 21,
                "vertebrae L5": 33,
                "vertebrae L4": 34,
                "vertebrae L3": 35,
                "vertebrae L2": 36,
                "vertebrae L1": 37,
                "vertebrae T12": 38,
                "vertebrae T11": 39,
                "vertebrae T10": 40,
                "vertebrae T9": 41,
                "vertebrae T8": 42,
                "vertebrae T7": 43,
                "vertebrae T6": 44,
                "vertebrae T5": 45,
                "vertebrae T4": 46,
                "vertebrae T3": 47,
                "vertebrae T2": 48,
                "vertebrae T1": 49,
                "vertebrae C7": 50,
                "vertebrae C6": 51,
                "vertebrae C5": 52,
                "vertebrae C4": 53,
                "vertebrae C3": 54,
                "vertebrae C2": 55,
                "vertebrae C1": 56,
                "skull": 120,
                "sternum": 122,
                "vertebrae S1": 127,
                "bone lesion": 128,
                "left rib 1": 63,
                "left rib 2": 64,
                "left rib 3": 65,
                "left rib 4": 66,
                "left rib 5": 67,
                "left rib 6": 68,
                "left rib 7": 69,
                "left rib 8": 70,
                "left rib 9": 71,
                "left rib 10": 72,
                "left rib 11": 73,
                "left rib 12": 74,
                "right rib 1": 75,
                "right rib 2": 76,
                "right rib 3": 77,
                "right rib 4": 78,
                "right rib 5": 79,
                "right rib 6": 80,
                "right rib 7": 81,
                "right rib 8": 82,
                "right rib 9": 83,
                "right rib 10": 84,
                "right rib 11": 85,
                "right rib 12": 86,
                "left humerus": 87,
                "right humerus": 88,
                "left scapula": 89,
                "right scapula": 90,
                "left clavicula": 91,
                "right clavicula": 92,
                "left femur": 93,
                "right femur": 94,
                "left hip": 95,
                "right hip": 96,
                "sacrum": 97,
                "costal cartilages": 114
            },
            "muscles": {
                "left gluteus maximus": 98,
                "right gluteus maximus": 99,
                "left gluteus medius": 100,
                "right gluteus medius": 101,
                "left gluteus minimus": 102,
                "right gluteus minimus": 103,
                "left autochthon": 104,
                "right autochthon": 105,
                "left iliopsoas": 106,
                "right iliopsoas": 107
            }
        }

    def label_id_to_name(self, label_id: int, label_dict: dict):
        """
        Get the label name from the label ID.

        Args:
            label_id: the label ID.
            label_dict: the label dictionary.
        """
        for group_dict in list(label_dict.values()):
            if isinstance(group_dict, dict):
                # this will skip str type value, such as "everything": <path>
                for label_name, label_id_ in group_dict.items():
                    if label_id == label_id_:
                        return label_name
        return None

    def segmentation_to_string(
        self,
        output_dir: Path,
        img_file: str,
        seg_file: str,
        label_groups: dict,
        modality: str = "CT",
        slice_index: int | None = None,
        axis: int = 2,
        image_filename: str = "image.jpg",
        label_filename: str = "label.jpg",
        output_prefix=None,
    ):
        """
        Format the segmentation response to a string.

        Args:
            response: the response.
            output_dir: the output directory.
            img_file: the image file path.
            modality: the modality.
            slice_index: the slice index.
            axis: the axis.
            image_filename: the image filename for the sliced image.
            label_filename: the label filename for the sliced image.
            group_label_names: the group label names to filter the label names.
            output_prefix: the output prefix.
            label_groups_path: the label groups path for VISTA-3D.
        """
        global SEGMENTATION_TOKEN
        output_dir = Path(output_dir)
        if output_prefix is None:
            output_prefix = f"The results are {SEGMENTATION_TOKEN}. The colors in this image describe "

        transforms = get_monai_transforms(
            ["image", "label"],
            output_dir,
            modality=modality,
            slice_index=slice_index,
            axis=axis,
            image_filename=image_filename,
            label_filename=label_filename,
        )
        data = transforms({"image": img_file, "label": seg_file})

        formatted_items = []

        for label_id in data["colormap"]:
            label_name = self.label_id_to_name(label_id, label_groups)
            if label_name is not None:
                color = data["colormap"][label_id]
                formatted_items.append(f"{color}: {label_name}")

        return output_prefix + ", ".join(formatted_items) + ". "

    def mentioned_by(self, input: str):
        """
        Check if the VISTA-3D model is mentioned in the input.

        Args:
            input (str): Text from the LLM, e.g. "Let me trigger <VISTA3D(arg)>."

        Returns:
            bool: True if the VISTA-3D model is mentioned, False otherwise.
        """
        matches = re.findall(r"<(.*?)>", str(input))
        if len(matches) != 1:
            return False
        return self.model_name in str(matches[0])

    def download_file(self, url: str, img_file: str):
        """
        Download the file from the URL.

        Args:
            url (str): The URL.
            img_file (str): The file path.
        """
        parent_dir = os.path.dirname(img_file)
        os.makedirs(parent_dir, exist_ok=True)
        with open(img_file, "wb") as f:
            response = requests.get(url)
            f.write(response.content)

    def run(
        self,
        img_file: str = "",
        image_url: str = "",
        input: str = "",
        output_dir: str = "",
        slice_index: int = 0,
        prompt: str = "",
        **kwargs,
    ):
        """
        Run the VISTA-3D model.

        Args:
            image_url (str): The image URL.
            input (str): The input text.
            output_dir (str): The output directory.
            img_file (str): The image file path. If not provided, download from the URL.
            slice_index (int): The slice index.
            prompt (str): The prompt text from the original request.
            **kwargs: Additional keyword arguments.
        """
        if not img_file:
            # Download from the URL
            img_file = os.path.join(output_dir, os.path.basename(image_url))
            self.download_file(image_url, img_file)

        output_dir = Path(output_dir)
        matches = re.findall(r"<(.*?)>", input)
        if len(matches) != 1:
            raise ValueError(f"Expert model {self.model_name} is not correctly enclosed in angle brackets.")

        match = matches[0]

        # Extract the arguments
        arg_matches = re.findall(r"\((.*?)\)", match[len(self.model_name) :])

        if len(arg_matches) == 0:  # <VISTA3D>
            arg_matches = ["everything"]
        if len(arg_matches) == 1 and (arg_matches[0] == "" or arg_matches[0] == None):  # <VISTA3D()>
            arg_matches = ["everything"]
        if len(arg_matches) > 1:
            raise ValueError(
                "Multiple expert model arguments are provided in the same prompt, "
                "which is not supported in this version."
            )

        vista3d_prompts = None
        label_groups = self._get_label_groups()

        if arg_matches[0] not in label_groups:
            raise ValueError(f"Label group {arg_matches[0]} is not accepted by the VISTA-3D model.")

        if arg_matches[0] != "everything":
            vista3d_prompts = [cls_idx for _, cls_idx in label_groups[arg_matches[0]].items()]

        # Trigger the VISTA-3D model
        input_dict = {"image": img_file}
        if vista3d_prompts is not None:
            input_dict["label_prompt"] = vista3d_prompts

        sys.path = [self.bundle_root] + sys.path

        with tempfile.TemporaryDirectory() as temp_dir:
            workflow = create_workflow(
                workflow_type="infer",
                bundle_root=self.bundle_root,
                config_file=os.path.join(self.bundle_root, f"configs/inference.json"),
                logging_file=os.path.join(self.bundle_root, "configs/logging.conf"),
                meta_file=os.path.join(self.bundle_root, "configs/metadata.json"),
                input_dict=input_dict,
                output_dtype="uint8",
                separate_folder=False,
                output_ext=".nii.gz",
                output_dir=temp_dir,
            )
            workflow.evaluator.run()
            output_file = os.path.join(temp_dir, os.listdir(temp_dir)[0])
            seg_file = os.path.join(output_dir, "segmentation.nii.gz")
            move(output_file, seg_file)

        seg_image = f"seg_{uuid4()}.jpg"
        text_output = self.segmentation_to_string(
            output_dir,
            img_file,
            seg_file,
            label_groups,
            modality="CT",
            slice_index=slice_index,
            image_filename=get_slice_filenames(img_file, slice_index),
            label_filename=seg_image,
        )

        if "segmented" in input:
            instruction = ""  # no need to ask for instruction
        else:
            instruction = "Use this result to respond to this prompt:\n" + prompt
        return text_output, os.path.join(output_dir, seg_image), instruction


class M3Generator:
    """Class to generate M3 responses"""

    def __init__(self, cache_images, source="huggingface", model_path="", conv_mode="", api_key="", experts_classes=[]):
        """Initialize the M3 generator"""
        global SYS_PROMPT
        self.cache_images = cache_images
        self.source = source
        self.experts_classes = experts_classes
        if source == "local" or source == "huggingface":
            from llava.conversation import conv_templates
            from llava.mm_utils import get_model_name_from_path
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init

            # Here we rewrite the global variable SYS_PROMPT
            # Since this class is initialized once in the demo
            # and the global variable will not updated after the initialization
            SYS_PROMPT = conv_templates[conv_mode].system

            # TODO: allow setting the device
            disable_torch_init()
            self.conv_mode = conv_mode
            if source == "huggingface":
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(model_path)
            model_name = get_model_name_from_path(model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path, model_name
            )
            logger.info(f"Model {model_name} loaded successfully. Context length: {self.context_len}")
        elif source == "nim":
            self.base_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/a2dec46a-b444-45aa-a1fc-a510ca41f186"
            if api_key == "":
                api_key = os.getenv("api_key", "Invalid")
            if api_key == "Invalid":
                raise ValueError("API key is not provided.")
            self.api_key = api_key
        elif source == "huggingface":
            pass
        else:
            raise NotImplementedError(f"Source {source} is not supported.")

    def generate_response_local(
        self,
        messages: list = [],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.9,
        system_prompt: str | None = None,
    ):
        """Generate the response"""
        logger.debug(f"Generating response with {len(messages)} messages")

        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token

        images = []

        conv = conv_templates[self.conv_mode].copy()
        if system_prompt is not None:
            conv.system = system_prompt
        user_role = conv.roles[0]
        assistant_role = conv.roles[1]

        for message in messages:
            role = user_role if message["role"] == "user" else assistant_role
            prompt = ""
            for content in message["content"]:
                if content["type"] == "text":
                    prompt += content["text"]
                if content["type"] == "image_path":
                    image_paths = (
                        content["image_path"] if isinstance(content["image_path"], list) else [content["image_path"]]
                    )
                    for image_path in image_paths:
                        images.append(load_image(image_path))
            conv.append_message(role, prompt)

        if conv.sep_style == SeparatorStyle.LLAMA_3:
            conv.append_message(assistant_role, "")  # add "" to the assistant message

        prompt_text = conv.get_prompt()
        logger.debug(f"Prompt input: {prompt_text}")

        if len(images) > 0:
            images_tensor = process_images(images, self.image_processor, self.model.config).to(
                self.model.device, dtype=torch.float16
            )
        images_input = [images_tensor] if len(images) > 0 else None

        tokens = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        input_ids = (tokens.unsqueeze(0).to(self.model.device))

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        start_time = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_input,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=1,
                max_new_tokens=max_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.eos_token_id,
                min_new_tokens=2,
            )
        end_time = time.time()
        logger.debug(f"Time taken to generate {len(output_ids[0])} tokens: {end_time - start_time:.2f} seconds")
        logger.debug(f"Tokens per second: {len(output_ids[0]) / (end_time - start_time):.2f}")

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        logger.debug(f"Assistant: {outputs}")

        return outputs

    def generate_response_nim(
        self,
        messages: list = [],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.9,
        **kwargs,
    ):
        """Generate the response using the NIM API"""
        logger.debug(f"Generating response with {len(messages)} messages using the NIM API")
        req_messages = []
        for message in messages:
            role = message["role"]  # expert has already been squashed into user
            contents = []
            for content in message["content"]:
                if content["type"] == "text":
                    contents.append({"type": "text", "text": content["text"]})
                if content["type"] == "image_path":
                    # if the path is cached from a URL, then use the URL
                    if content["image_path"] in self.cache_images.cache_images.values():
                        for url, value in self.cache_images.cache_images.items():
                            if value == content["image_path"]:
                                local_path = self.cache_images.dir()
                                url = url.replace(local_path, REMOTE_URL)
                                contents.append({"type": "image_url", "image_url":{"url": url}})
                    elif os.path.exists(content["image_path"]):
                        data_url = image_to_data_url(content["image_path"], max_size=(384, 384))
                        logger.debug(f"Length of the data URL: {len(data_url)}")
                        contents.append({"type": "image_url", "image_url": {"url": data_url}})
            req_messages.append({"role": role, "content": contents})
        logger.debug(f"Request messages: {req_messages}")
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            },
            json={
                "messages": req_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        )

        try:
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to get the response from the NIM API: {e}")
            return f"Sorry, I met an error: {e}."

    def generate_response(self, **kwargs):
        """Generate the response"""
        if self.source == "local" or self.source == "huggingface":
            return self.generate_response_local(**kwargs)
        elif self.source == "nim":
            raise NotImplementedError("NIM API is not supported in the local demo.")
            # return self.generate_response_nim(**kwargs)
        raise NotImplementedError(f"Source {self.source} is not supported.")

    def squash_expert_messages_into_user(self, messages: list):
        """Squash consecutive expert messages into a single user message."""
        logger.debug("Squashing expert messages into user messages")
        messages = deepcopy(messages)  # Create a deep copy to avoid modifying the original list

        i = 0
        while i < len(messages):
            if messages[i]["role"] == "expert":
                messages[i]["role"] = "user"
                j = i + 1
                while j < len(messages) and messages[j]["role"] == "expert":
                    messages[i]["content"].extend(messages[j]["content"])  # Append the content directly
                    j += 1
                del messages[i + 1 : j]  # Remove all the squashed expert messages

            i += 1

        return messages

    def process_prompt(self, prompt, sv, chat_history):
        """Process the prompt and return the result. Inputs/outputs are the gradio components."""
        logger.debug(f"Process the image and return the result")

        if sv.temp_working_dir is None:
            sv.temp_working_dir = tempfile.mkdtemp()

        if sv.modality_prompt == "Auto":
            modality = get_modality(sv.image_url, text=prompt)
        else:
            modality = sv.modality_prompt
        mod_msg = f"This is a {modality} image.\n" if modality != "Unknown" else ""

        model_cards = sv.sys_msg if sv.use_model_cards else ""

        img_file = self.cache_images.get(sv.image_url, None, list_return=True)

        if isinstance(img_file, str):
            if "<image>" not in prompt:
                _prompt = model_cards + "<image>" + mod_msg + prompt
                sv.sys_msgs_to_hide.append(model_cards + "<image>" + mod_msg)
            else:
                _prompt = model_cards + mod_msg + prompt
                if model_cards + mod_msg != "":
                    sv.sys_msgs_to_hide.append(model_cards + mod_msg)

            if img_file.endswith(".nii.gz"):  # Take the specific slice from a volume
                chat_history.append(
                    _prompt,
                    image_path=os.path.join(self.cache_images.dir(), get_slice_filenames(img_file, sv.slice_index)),
                )
            else:
                chat_history.append(_prompt, image_path=img_file)
        elif isinstance(img_file, list):
            # multi-modal images
            prompt = (
                prompt.replace("<image>", "") if "<image>" in prompt else prompt
            )  # remove the image token if it's in the prompt
            special_token = "T1(contrast enhanced): <image1>, T1: <image2>, T2: <image3>, FLAIR: <image4> "
            mod_msg = f"These are different {modality} modalities.\n"
            _prompt = model_cards + special_token + mod_msg + prompt
            image_paths = [os.path.join(self.cache_images.dir(), get_slice_filenames(f, sv.slice_index)) for f in img_file]
            chat_history.append(_prompt, image_path=image_paths)
            sv.sys_msgs_to_hide.append(model_cards + special_token + mod_msg)
        elif img_file is None:
            # text-only prompt
            chat_history.append(prompt)  # no image token
        else:
            raise ValueError(f"Invalid image file: {img_file}")

        logger.info(f"Processing the prompt: {prompt}, with max tokens: {sv.max_tokens}, temperature: {sv.temperature}, top P: {sv.top_p}, slice index: {sv.slice_index}")
        outputs = self.generate_response(
            messages=self.squash_expert_messages_into_user(chat_history.messages),
            max_tokens=sv.max_tokens,
            temperature=sv.temperature,
            top_p=sv.top_p,
            system_prompt=sv.sys_prompt,
        )

        chat_history.append(outputs, role="assistant")

        # check the message mentions any expert model
        expert = None

        for expert_model in self.experts_classes:
            expert = expert_model() if expert_model().mentioned_by(outputs) else None
            if expert:
                break

        if expert:
            logger.info(f"Expert model {expert.__class__.__name__} is being called to process {sv.image_url}.")
            try:
                if sv.image_url is None:
                    logger.debug(
                        "Image URL is None. Try restoring the image URL from the backup to continue expert processing."
                    )
                    sv.restore_from_backup("image_url")
                    sv.restore_from_backup("slice_index")
                text_output, seg_image, instruction = expert.run(
                    image_url=sv.image_url,
                    input=outputs,
                    output_dir=sv.temp_working_dir,
                    img_file=self.cache_images.get(sv.image_url, None, list_return=True),
                    slice_index=sv.slice_index,
                    prompt=prompt,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.debug(f"Error: {e}")
                text_output = f"Sorry I met an error: {e}"
                seg_image = None
                instruction = ""

            chat_history.append(text_output, image_path=seg_image, role="expert")
            if instruction:
                chat_history.append(instruction, role="expert")
                outputs = self.generate_response(
                    messages=self.squash_expert_messages_into_user(chat_history.messages),
                    max_tokens=sv.max_tokens,
                    temperature=sv.temperature,
                    top_p=sv.top_p,
                    system_prompt=sv.sys_prompt,
                )
                chat_history.append(outputs, role="assistant")

        new_sv = SessionVariables()
        # Keep these parameters accross one conversation
        new_sv.sys_prompt=sv.sys_prompt
        new_sv.sys_msg=sv.sys_msg
        new_sv.use_model_cards=sv.use_model_cards
        new_sv.temp_working_dir=sv.temp_working_dir
        new_sv.max_tokens=sv.max_tokens
        new_sv.temperature=sv.temperature
        new_sv.top_p=sv.top_p
        # new_sv.interactive=True,
        new_sv.sys_msgs_to_hide=sv.sys_msgs_to_hide
        new_sv.backup={"image_url": sv.image_url, "slice_index": sv.slice_index},

        return (
            new_sv,
            chat_history,
        )
