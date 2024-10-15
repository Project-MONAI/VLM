import base64
import itertools
import json
import logging
import os
import re
from io import BytesIO
from pathlib import Path
from shutil import move
from typing import List
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np
import requests
import skimage
from monai.transforms import (Compose, LoadImageD, MapTransform, OrientationD,
                              ScaleIntensityD, ScaleIntensityRangeD)
from PIL import Image as PILImage
from PIL.Image import Image
from schemas import ModelCard

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


def classification_to_string(outputs):
    """Format the classification outputs to a string."""
    def binary_output(value):
        return "yes" if value >= 0.5 else "no"

    def score_output(value):
        return f"{value:.2f}"

    formatted_items = [
        f"{key.lower().replace('_', ' ')}: {binary_output(outputs[key])}"
        for key in sorted(outputs)
    ]

    return "\n".join(
        ["The resulting predictions are:"] + formatted_items + ["."]
    )


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


def label_id_to_name(label_id: int, label_dict: dict, group_label_names: list | str | None = None):
    """
    Get the label name from the label ID.

    Args:
        label_id: the label ID.
        label_dict: the label dictionary.
        group_label_names: the group label names to filter the label names.
    """
    if group_label_names is None:
        group = list(label_dict.values())
    elif isinstance(group_label_names, list):
        group = [label_dict[group_name] for group_name in group_label_names]
    elif isinstance(group_label_names, str):
        group = [label_dict[group_label_names]]
    else:
        raise ValueError("group_label_names must be a list of strings or a single string.")

    for group_dict in group:
        if isinstance(group_dict, dict):
            # this will skip str type value, such as "everything": <path>
            for label_name, label_id_ in group_dict.items():
                if label_id == label_id_:
                    return label_name
    return None


def save_image_url_to_file(
        image_url: str,
        output_dir: Path,
        allow_redirects: bool = True,
        timeout: int | tuple[int, int] = 10,
        verify: bool = True,
        domain_whitelist: list[str] = [".*"],
        domain_blacklist: list[str] = [""],
    ):
    """
    Save the image from the URL to the file.

    Args:
        image_url: the image URL.
        output_dir: the output directory.
        allow_redirects: allow redirects.
        timeout: the timeout for the request.
        verify: verify the SSL certificate.
        domain_whitelist: the domain whitelist.
        domain_blacklist: the domain blacklist.
    
    Raises:
        requests.exceptions.RequestException: Error fetching image.
        requests.exceptions.RequestException: URL domain not allowed.
        requests.exceptions.RequestException: File size too large.
    """
    logger.debug(f"Saving image to {output_dir}")
    output_dir = Path(output_dir)

    try:
        url_response = requests.get(
            image_url,
            allow_redirects=allow_redirects,
            timeout=timeout,
            verify=verify,
        )
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException("Error fetching image")

    if url_response.status_code != 200:
        raise requests.exceptions.RequestException(f"Error fetching image. Status code: {url_response.status_code}")

    if not is_url_allowed_domain(image_url, domain_whitelist, domain_blacklist):
        raise requests.exceptions.RequestException("Error fetching image. URL domain not allowed")

    content_disposition = url_response.headers.get("Content-Disposition")

    # TODO: check the what file types are allowed
    # content_type = url_response.headers.get("Content-Type")

    if int(url_response.headers.get("Content-Length", 0)) > 1000000000:
        raise requests.exceptions.RequestException("Error fetching image. File size too large")

    file_name = output_dir / get_filename_from_cd(image_url, content_disposition)
    with open(file_name, "wb") as f:
        f.write(url_response.content)

    return file_name


def save_zipped_seg_to_file(
        zip_response: requests.Response,
        output_dir: Path,
        output_name: str = "segmentation",
        output_ext: str = ".nrrd"
    ):
    """
    Save the segmentation file from the zip response to the file.

    Args:
        zip_response: the zip response.
        output_dir: the output directory.
        output_name: the output name.
        output_ext: the output extension.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)
    with ZipFile(BytesIO(zip_response.content)) as zip_file:
        zip_file.extractall(output_dir)

    file_list = os.listdir(output_dir)
    for f in file_list:
        f = Path(f)
        file_path = output_dir / f
        if file_path.exists() and f.suffix == output_ext:
            move(file_path, output_dir / f"{output_name}{output_ext}")
            return output_dir / f"{output_name}{output_ext}"

    raise FileNotFoundError(f"Segmentation file not found in {output_dir}")


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


def segmentation_to_string(
        output_dir: Path,
        img_file: str,
        seg_file: str,
        modality: str = "CT",
        slice_index: int | None = None,
        axis: int = 2,
        image_filename: str = "image.jpg",
        label_filename: str = "label.jpg",
        group_label_names: str | list | None = None,
        output_prefix = "The results are <segmentation>. The colors in this image describe ",
        label_groups_path: str = "label_groups_dict.json",
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
        group_label_names: the group label names to filter the label names.
        output_prefix: the output prefix.
        label_groups_path: the label groups path for VISTA-3D.
    """
    output_dir = Path(output_dir)
    
    transforms = get_monai_transforms(
        ["image", "label"],
        output_dir,
        modality=modality,
        slice_index=slice_index,
        axis=axis,
    )
    data = transforms({"image": img_file, "label": seg_file})

    with open(label_groups_path) as f:
        label_groups = json.load(f)

    formatted_items = []

    for label_id in data["colormap"]:
        label_name = label_id_to_name(label_id, label_groups, group_label_names=group_label_names)
        if label_name is not None:
            color = data["colormap"][label_id]
            formatted_items.append(f"{color}: {label_name}")

    return output_prefix + ", ".join(formatted_items) + ". "


def trigger_expert_endpoint(
        endpoint: str,
        image_url: str,
        arg: str | None = None,
        label_groups_path: str = "label_groups_dict.json",
    ):
    """
    Trigger the expert model endpoint.

    Args:
        endpoint: the endpoint.
        image_url: the image URL.
        arg: the argument.
        label_groups_path: the label groups path for VISTA-3D.
    """
    logger.debug(f"Triggering expert model endpoint: {endpoint}")
    payload = {"image": image_url}

    if arg is not None and arg != "everything":
        # VISTA-3D specific logics
        with open(label_groups_path) as f:
            label_groups = json.load(f)

        if arg not in label_groups:
            raise ValueError(f"Label group {arg} not found in {label_groups_path}")

        payload["prompts"] = {"classes": list(label_groups[arg].keys())}

    api_key = None
    if "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions" in endpoint:
        api_key = os.getenv("NVCF_INTERNAL_KEY", "Invalid")
    elif "https://health.api.nvidia.com" in endpoint:
        api_key = os.getenv("NIM_API_KEY", "Invalid")

    if api_key == "Invalid":
        raise ValueError(f"Expert model API key not found to trigger {endpoint}")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "accept": "application/json",
    } if api_key is not None else {}

    response = requests.post(endpoint, headers=headers, json=payload)

    if response.status_code != 200:
        payload.pop("image")  # hide the image URL in the error message
        raise requests.exceptions.HTTPError(f"Error triggering POST to {endpoint} with Payload {payload}: {response.status_code}")
    return response


def parse_user_input_messages(messages: List[Message]):
    """Get the last user prompt sent with an image"""
    # reversely iterate the messages to get the last user message that has "image_url"
    # Initialize variables
    prompt_parts = []
    image_url = None
    slice_index = None
    axis = None

    # Iterate over messages in reverse to find the latest user message with an image
    for message in reversed(messages):
        if message.role == "user" and isinstance(message.content, list):
            for content in message.content:
                if content.type == "text":
                    prompt_parts.append(content.text)
                elif content.type == "image_url":
                    image_url = content.image_url.url
                    slice_index = content.image_url.slice_index
                    axis = content.image_url.axis

            # Break if image_url is found
            if image_url:
                break

    # Join prompt text parts in reverse to preserve original order
    prompt = ''.join(reversed(prompt_parts))

    return prompt, image_url, slice_index, 2 if axis is None else axis


def get_system_messge(model_cards: List[ModelCard]):
    """Get the system message with the model cards"""
    system_message = "Here is a list of available expert models:\n"
    for model in model_cards:
        model_valid_args = None
        if isinstance(model.valid_args, list):
            model_valid_args = "'" + "', '".join(model.valid_args[:-1]) + "'"
            if len(model.valid_args) > 1:
                model_valid_args += f", or '{model.valid_args[-1]}'"
        system_message += f"<{model.name}(args)> {model.description}, Valid args are: {model_valid_args}\n"
    system_message += "Select the most suitable expert model to answer the prompt and give the model <NAME(args)>.\n"
    return system_message


def insert_system_message(messages: List[Message], model_cards: List[ModelCard], modality: str = "Unknown"):
    """Insert the system message to the messages"""
    sys_msg = get_system_messge(model_cards)
    for message in messages:
        # Be aware that the expert role also has `image_url` for segmentation
        if message.role == "user" and isinstance(message.content, list):
            has_image = [c for c in message.content if c.type == "image_url"]
            if has_image:
                url = has_image[0].image_url.url  # TODO: support multiple images
                _modality = modality if modality != "Unknown" else get_modality(url)
                for i, content in enumerate(message.content):
                    if content.type == "text":
                        if _modality != "Unknown":
                            # Below commented line is from the training sample. However, it doesn't work.
                            # content.text = sys_msg + f"<image> This is a {modality} image.\n" + content.text
                            # So, we have to change it to the following line.
                            content.text = sys_msg + f"This is a {_modality} image.\n" + content.text
                        else:
                            content.text = sys_msg + content.text
                        if "<image>" not in content.text:
                            content.text += " <image>"
                        message.content[i].text = content.text
                        break  # Only apply the system message once per user message
    return messages



def load_image(image_url: str) -> Image:
    """
    Load the image from the URL.

    Args:
        image_url: the image URL.
        prompt: the prompt that goes with the image, which can be used to determine the modality.
        slice_index: the slice index for a 3D image. Will not be used for 2D images.
        axis: the axis for a 3D image. Will not be used for 2D images.
    """
    logger.debug(f"Loading image from URL")
    if os.path.exists(image_url) and is_url_allowed_file_type(image_url, None, [".png", ".jpg", ".jpeg"]):
        image = PILImage.open(image_url).convert("RGB")
    elif image_url.startswith("http") or image_url.startswith("https"):
        if is_url_allowed_file_type(image_url, None, [".png", ".jpg", ".jpeg"]):
            response = requests.get(image_url)
            image = PILImage.open(BytesIO(response.content)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {image_url}")
    else:
        image_base64_regex = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")
        match_results = image_base64_regex.match(image_url)
        if match_results is None:
            raise ValueError(f"Invalid image url: {image_url}")
        image_base64 = match_results.groups()[1]
        image = PILImage.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
    return image


def check_early_stop(prompt: str, model_card: ModelCard):
    """
    Check if the prompt contains a stop word "segmented" to early stop the segmentation model.

    Args:
        prompt: the prompt.
        model_card: the model card.
    """
    if model_card.task == "segmentation" and "segmented" in prompt.lower():
        return True
    return False


def take_slice_and_insert_user_messages(
        messages: List[Message],
        prompt: str,
        image_path: str,
        image_url: str,
        slice_index: int | None,
        axis: int | None,
        output_dir: Path,
    ):
    """
    Get a 2D slice from a 3D image URL and replace the image URL in the messages.

    Returns:
        The updated messages and the modality of the image.
    """
    modality = "Unknown"
    if not is_url_allowed_file_type(image_url, None, SUPPORTED_3D_IMAGE_FORMATS):
        # If the image URL is not a 3D image, return the messages as is
        return messages, modality

    if not os.path.exists(image_path):
        # the image is not saved to the file yet
        return messages, modality

    for message in messages:
        if message.role == "user" and isinstance(message.content, list):
            located_index = None
            for i, content in enumerate(message.content):
                if content.type == "image_url" and content.image_url.url == image_url:
                    located_index = i
                    break  # stop iterating contents if image_url is found

            if located_index:
                modality = get_modality(image_url, text=prompt)
                transforms = get_monai_transforms(
                    ["image"],
                    Path(output_dir),
                    modality=modality,
                    slice_index=slice_index,
                    axis=axis,
                )
                transforms({"image": image_path})
                message.content[located_index].image_url.url = os.path.join(output_dir, "image.jpg")
                break  # stop iterating messages after the url replacement

    return messages, modality


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
