import itertools
import logging
import os
import re

from pathlib import Path
from typing import List
from urllib.parse import urlparse
from PIL import Image

import numpy as np
import requests
import skimage
from monai.transforms import (Compose, LoadImageD, MapTransform, OrientationD,
                              ScaleIntensityD, ScaleIntensityRangeD)


logger = logging.getLogger('uvicorn.error')


MODALITY_MAP = {
    "cxr": "CXR",
    "chest x-ray": "CXR",
    "ct image": "CT",
    "mri": "MRI",
    "magnetic resonance imaging": "MRI",
    "ultrasound": "US",
    "cell imaging": "cell imaging",
}


SUPPORTED_3D_IMAGE_FORMATS = [".nii", ".nii.gz", ".nrrd"]


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
        d = dict(data)
        for key in self.key_iterator(d):
            np_array = np.squeeze(d.get(key))
            slice_index = np_array.shape[2] // 2  if self.slice_index is None else self.slice_index
            slice = np.take(np_array, slice_index, axis=self.axis)
            d[key] = np.rot90(np.swapaxes(slice.astype(np.uint8), 0, 1), k=2)

        os.makedirs(self.output_dir, exist_ok=True)
        skimage.io.imsave(self.output_dir / self.image_filename, np.stack([d[self.image_key]] * 3, axis=-1))

        if self.label_key in d:
            color_label = skimage.color.label2rgb(
                d[self.label_key],
                colors=self.COLORS,
                image=d[self.image_key],
                bg_label=self.bg_label
            ) * 255

            skimage.io.imsave(self.output_dir / self.label_filename, color_label.astype(np.uint8))

            unique_labels = np.unique(d[self.label_key])
            color_cyle = itertools.cycle(Dye.COLORS)

            colormap = {}
            unique_labels = unique_labels[unique_labels != self.bg_label]  # remove background label
            for label_id, label_color in zip(unique_labels, color_cyle):
                colormap[label_id] = label_color
            d["colormap"] = colormap
        return d


def get_filename_from_cd(url, cd):
    """Get filename from content-disposition"""
    if not cd:
        if url.find("/"):
            return url.rsplit("/", 1)[1]
        return None

    fname = re.findall("filename=(.+)", cd)
    if len(fname) == 0:
        return None
    return fname[0].strip('"').strip("'")


def get_slice_filenames(image_file, slice_index):
    """Small helper function to get the slice filenames"""
    base_name = os.path.basename(image_file)
    image_filename = base_name.replace(".nii.gz", f"_slice{slice_index}.jpg")
    seg_filename = base_name.replace(".nii.gz", f"_slice{slice_index}_seg.jpg")
    return image_filename, seg_filename


def is_url(url):
    """Function to check if the URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


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
            return PILImage.open(image_path_or_data_url).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load the image: {e}")
    else:
        image_base64_regex = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")
        match_results = image_base64_regex.match(image_path_or_data_url)
        if match_results:
            image_base64 = match_results.groups()[1]
            return PILImage.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")

    raise ValueError(f"Unable to load the image from {image_path_or_data_url[:50]}")


def is_url_allowed_file_type(url: str, content_disposition: str | None, supported_type: List[str]) -> bool:
    """Function to check if the URL's extension or Content-Disposition indicates an allowed file type"""
    # Check based on URL extension
    for ext in supported_type:
        if url.endswith(ext):
            return True

    # Extract filename from Content-Disposition and check its extension
    if content_disposition:
        filename = re.findall('filename="(.+)"', content_disposition)
        if filename and any(filename[0].endswith(ext) for ext in supported_type):
            return True

    return False


def is_url_allowed_domain(url: str, whitelist: List[str], blacklist: List[str]) -> bool:
    """Function to check URL against whitelist and blacklist"""
    if not is_url(url):
        return False
    if blacklist != [""]:
        if any(re.match(pattern, url) for pattern in blacklist):
            return False
    return any(re.match(pattern, url) for pattern in whitelist)


def _get_modality_url(image_url: str | None):
    """Hardcoded prompt based on the file path"""
    if not isinstance(image_url, str):
        return "Unknown"
    if image_url.startswith("data:image"):
        return "Unknown"
    if ".nii.gz" in image_url.lower():
        return "CT"
    if "cxr_" in image_url.lower():
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


def get_monai_transforms(
        keys,
        output_dir: Path | str,
        image_key = "image",
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
            a_min=window_center-window_width/2,
            a_max=window_center+window_width/2,
            b_min=0,
            b_max=255,
            clip=True
        )
    elif modality == "MRI":
        scaler = ScaleIntensityD(
            keys=[image_key],
            minv=0,
            maxv=255,
            channel_wise=True
        )
    else:
        raise ValueError(f"Unsupported modality: {modality}. Supported modalities are 'CT' and 'MRI'.")

    return Compose([
        LoadImageD(keys=keys, ensure_channel_first=True),
        OrientationD(keys=keys, axcodes="RAS"),
        scaler,
        Dye(slice_index=slice_index, axis=axis, output_dir=output_dir, image_filename=image_filename, label_filename=label_filename),
    ])


def manage_errors(e: Exception):
    """Manage the errors and return the error message"""
    bot_liked_output = "I'm sorry I can't continue, because "
    unhandled_error = ""
    match e:
        case requests.exceptions.RequestException:
            if "Error fetching image" in str(e):
                # append a message to the messages list and returns the response
                bot_liked_output += "I am unable to reach the given URL"
            else:
                unhandled_error = str(e)
        case ValueError:
            if "Invalid expert model" in str(e):
                bot_liked_output += "I am unable to find the info for the suitable model card. Please provide more context."
            elif "Multiple expert models" in str(e):
                bot_liked_output += "I found multiple expert model cards and cannot continue. Please provide more context."
            elif "Error triggering POST" in str(e):
                bot_liked_output += (
                    "I had an error when I tried to trigger POST request to the expert model endpoint. "
                    "Please try selecting different expert models to help me understand the problem."
                )
            else:
                unhandled_error = str(e)
    return bot_liked_output, unhandled_error
