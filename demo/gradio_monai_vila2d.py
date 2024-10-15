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

import argparse
import base64
import html
import logging
import os
import json
import shutil
import tempfile
import urllib.request
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
import re

import argparse
import base64
import logging
import os
import shutil
import tempfile
import traceback
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy

import torch
from dotenv import load_dotenv
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from utils import (check_early_stop, classification_to_string, get_slice_filenames,
                   get_modality, get_model_card_and_arg, insert_system_message,
                   load_image, manage_errors, parse_user_input_messages,
                   get_filename_from_cd, segmentation_to_string,
                   take_slice_and_insert_user_messages, save_zipped_seg_to_file,
                   trigger_expert_endpoint, get_monai_transforms)

import gradio as gr
import nibabel as nib
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)

# Suppress logging from dependent libraries
logging.getLogger("gradio").setLevel(logging.WARNING)

# Sample images dictionary
IMAGES_URLS = {
    "CT Sample 1": "https://developer.download.nvidia.com/assets/Clara/monai/samples/liver_0.nii.gz",
    "Chest X-ray Sample 1": "https://developer.download.nvidia.com/assets/Clara/monai/samples/cxr_ce3d3d98-bf5170fa-8e962da1-97422442-6653c48a_v1.jpg",
    "Chest X-ray Sample 2": "https://developer.download.nvidia.com/assets/Clara/monai/samples/cxr_fcb77615-ceca521c-c8e4d028-0d294832-b97b7d77_v1.jpg",
    "Chest X-ray Sample 3": "https://developer.download.nvidia.com/assets/Clara/monai/samples/cxr_6cbf5aa1-71de2d2b-96f6b460-24227d6e-6e7a7e1d_v1.jpg",
}

HARDCODED_EXPERT_MODELS = ["VISTA3D", "CXR"]

MODEL_CARDS = [
    # {
    #     "name": "BRATS",
    #     "task": "segmentation",
    #     "description": (
    #         "Modality: MRI, "
    #         "Task: segmentation, "
    #         "Overview: A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data, "
    #         "Accuracy: Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 - Average: 0.8518"
    #     ),
    #     "valid_args": None,
    #     "endpoint": os.getenv("BRATS_ENDPOINT", ""),
    #     "execution": None,
    # },
    {
        "name": "VISTA3D",
        "task": "segmentation",
        "description": (
            "Modality: CT, "
            "Task: segmentation, "
            "Overview: domain-specialized interactive foundation model developed for segmenting and annotating human anatomies with precision, "
            "Accuracy: 127 organs: 0.792 Dice on average"
        ),
        "valid_args": ["everything", "hepatic tumor", "pancreatic tumor", "lung tumor", "bone lesion", "organs", "cardiovascular", "gastrointestinal", "skeleton", "muscles"],
        "endpoint": os.getenv("V3D_ENDPOINT", "https://health.api.nvidia.com/v1/medicalimaging/nvidia/vista-3d"),
        "execution": "endpoint"
    },
    # {
    #     "name": "VISTA2D",
    #     "task": "segmentation",
    #     "description": (
    #         "Modality: cell imaging, "
    #         "Task: segmentation, "
    #         "Overview: model for cell segmentation, which was trained on a variety of cell imaging outputs, including brightfield, phase-contrast, fluorescence, confocal, or electron microscopy, "
    #         "Accuracy: Good accuracy across several cell imaging datasets"
    #     ),
    #     "valid_args": None,
    #     "endpoint": os.getenv("V2D_ENDPOINT", ""),
    # },
    {
        "name": "CXR",
        "task": "classification",
        "description": (
            "Modality: chest x-ray (CXR), "
            "Task: classification, "
            "Overview: pre-trained model which are trained on large cohorts of data, "
            "Accuracy: Good accuracy across several diverse chest x-rays datasets"
        ),
        "valid_args": None,
        "endpoint": os.getenv("CXR_ENDPOINT", "http://dlmed-api-m3.nvidia.com:7998/torchxrayvision/run"),
    },
]


EXAMPLE_PROMPTS = [
    "Segment the visceral structures in the current image.",
    "Can you identify any liver masses or tumors?",
    "Segment the entire image.",
    "What abnormalities are seen in this image?",
    "Is there evidence of edema in this image?",
    "Is there evidence of any abnormalities in this image?",
    "What is the total number of [condition/abnormality] present in this image?",
    "Is there pneumothorax?",
    "What type is the lung opacity?",
    "which view is this image taken?",
    "Is there evidence of cardiomegaly in this image?",
    "Is the atelectasis located on the left side or right side?",
    "What level is the cardiomegaly?",
]

HTML_PLACEHOLDER = "<br>".join([""] * 15)

CACHED_DIR = tempfile.mkdtemp()

CACHED_IMAGES = {}

TITLE = """
    <div style="text-align: center; max-width: 650px; margin: 0 auto;">
        <p>
        <img src="https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/MONAI-logo-color.png" alt="project monai" style="width: 50%; min-width: 500px; max-width: 800px; margin: auto; display: block;">
        </p>
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-bottom: 7px;">
            MONAI Multi-Modal Medical (M3) VLM Demo
        </h1>
        </div>
        <p style="margin-bottom: 10px; font-size: 94%">
        Placeholder text for the description of the tool.
        </p>

    </div>
"""

CSS_STYLES = (
    ".fixed-size-image {\n",
    "width: 512px;\n",
    "height: 512px;\n",
    "object-fit: cover;\n",
    "}\n",
    ".small-text {\n",
    "font-size: 6px;\n",
    "}\n",
)


def cache_images():
    """Cache the image and return the file path"""
    logger.debug(f"Caching the image")
    for _, image_url in IMAGES_URLS.items():
        try:
            url_response = requests.get(image_url, allow_redirects=True)
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Failed to download the image: {e}")

        if url_response.status_code != 200:
            raise requests.exceptions.RequestException(f"Failed to download the image: {e}")

        content_disposition = url_response.headers.get("Content-Disposition")
        CACHED_IMAGES[image_url] = os.path.join(CACHED_DIR, get_filename_from_cd(image_url, content_disposition))


def cache_cleanup():
    """Clean up the cache"""
    logger.debug(f"Cleaning up the cache")
    for _, cache_file_name in CACHED_IMAGES.items():
        if os.path.exists(cache_file_name):
            os.remove(cache_file_name)
            print(f"Cache file {cache_file_name} cleaned up")


def input_image(image, sv):
    """Update the session variables with the input image data URL if it's inputted by the user"""
    logger.debug(f"Received user input image")
    # TODO: support user uploaded images
    sv.image_url = image_to_data_url(image)
    return image, sv


def update_image_selection(selected_image, sv, slice_index_html, increment=None):
    """Update the gradio components based on the selected image"""
    logger.debug(f"Updating display image for {selected_image}")
    sv.image_url = IMAGES_URLS.get(selected_image, None)
    img_file = CACHED_IMAGES.get(sv.image_url, None)

    if sv.image_url is None:
        return None, sv, slice_index_html

    if sv.temp_working_dir is None:
        sv.temp_working_dir = tempfile.mkdtemp()

    if img_file.endswith(".nii.gz"):
        if sv.slice_index is None:
            data = nib.load(img_file).get_fdata()
            sv.slice_index = data.shape[sv.axis] // 2
            sv.idx_range = (0, data.shape[sv.axis] - 1)

        if increment is not None:
            sv.slice_index += increment
            sv.slice_index = max(sv.idx_range[0], min(sv.idx_range[1], sv.slice_index))

        image_filename = get_slice_filenames(img_file, sv.slice_index)[0]
        if not os.path.exists(image_filename):
            transforms = get_monai_transforms(
                    ["image"],
                    sv.temp_working_dir,
                    modality="CT",  # TODO: Get the modality from the image/prompt/metadata
                    slice_index=sv.slice_index,
                    image_filename=image_filename,
                )
            transforms({"image": img_file})
        return image_filename, sv, f"Slice Index: {sv.slice_index}"

    sv.slice_index = None    
    return (
        img_file,
        sv,
        "Slice Index: N/A for 2D images, clicking prev/next will not change the image.",
    )


def update_image_next_10(selected_image, sv, slice_index_html):
    """Update the image to the next 10 slices"""
    return update_image_selection(selected_image, sv, slice_index_html, increment=10)


def update_image_next_1(selected_image, sv, slice_index_html):
    """Update the image to the next slice"""
    return update_image_selection(selected_image, sv, slice_index_html, increment=1)


def update_image_prev_1(selected_image, sv, slice_index_html):
    """Update the image to the previous slice"""
    return update_image_selection(selected_image, sv, slice_index_html, increment=-1)


def update_image_prev_10(selected_image, sv, slice_index_html):
    """Update the image to the previous 10 slices"""
    return update_image_selection(selected_image, sv, slice_index_html, increment=-10)


def image_to_data_url(image, format="JPEG", max_size=None):
    """
    Convert an image to a data URL.

    Args:
        image (str | np.Array): The image to convert. If a string, it is treated as a file path.
        format (str): The format to save the image in. Default is "JPEG".
        max_size (tuple): The maximum size of the image. Default is None.
    """
    logger.debug(f"Converting image to data URL")
    if isinstance(image, str) and image.startswith("data:image"):
        return image
    if isinstance(image, str) and image.startswith("http"):
        logger.debug(f"Received Image URL: {image}")
        img = Image.open(requests.get(image, stream=True).raw)
    elif isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
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
    if len(img_base64) > 180_000:
        logger.warning(
            (
                f"The image is too large for the data URL. "
                "Use the assets API or use the following snippet to resize:\n {IMAGE_SIZE_WARNING}."
            )
        )
    # Convert the base64 bytes to string and format the data URL
    return f"data:image/{format.lower()};base64,{img_base64}"


def resize_data_url(data_url, max_size):
    """
    Resize a data URL image to a maximum size.

    Args:
        data_url (str): The data URL of the image.
        max_size (tuple): The maximum size of the image.
    """
    logger.debug(f"Resizing data URL image")
    # Convert the data URL to an image
    img = Image.open(BytesIO(base64.b64decode(data_url.split(",")[1])))
    # Resize the image
    img.thumbnail(max_size)
    # Create a BytesIO buffer to save the image
    buffered = BytesIO()
    # Save the image to the buffer in the specified format
    img.save(buffered, format="JPEG")
    # Convert the buffer content into bytes
    img_byte = buffered.getvalue()
    # Encode the bytes to base64
    img_base64 = base64.b64encode(img_byte).decode()
    # Convert the base64 bytes to string and format the data URL
    return f"data:image/jpeg;base64,{img_base64}"


def colorcode_message(text="", data_url=None, show_all=False, role="user"):
    """Color the text based on the role and return the HTML text"""
    logger.debug(f"Preparing the HTML text with {show_all} and role: {role}")
    # if content is not a data URL, escape the text

    if not show_all and role == "expert":
        return ""
    escaped_text = html.escape(text)
    if data_url is not None:
        escaped_text += f'<img src="{data_url}">'
    if role == "user":
        return f'<p style="color: blue;">User:</p> {escaped_text}'
    elif role == "expert":
        return f'<p style="color: green;">Expert:</p> {escaped_text}'
    elif role == "assistant":
        return f'<p style="color: red;">AI Assistant:</p> {escaped_text}</p>'
    raise ValueError(f"Invalid role: {role}")


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

    def get_html(self, show_all=False):
        """Returns the chat history as an HTML string to display"""
        history = []

        for message in self.messages:
            role = message["role"]
            contents = message["content"]
            history_text_html = ""
            for content in contents:
                if content["type"] == "text":
                    history_text_html += colorcode_message(text=content["text"], show_all=show_all, role=role)
                else:
                    history_text_html += colorcode_message(
                        data_url=image_to_data_url(content["image_path"], max_size=(300, 300)),
                        show_all=True,
                        role=role
                    )  # always show the image
            history.append(history_text_html)
        return "<br>".join(history)


class SessionVariables:
    """Class to store the session variables"""
    def __init__(self):
        """Initialize the session variables"""
        self.model_cards = deepcopy(MODEL_CARDS)
        self.expert_models = [m["name"] for m in self.model_cards]
        self.slice_index = None  # Slice index for 3D images
        self.image_path = None  # Image path to display and process
        self.top_p = 0.9 
        self.temperature = 0.0
        self.max_tokens = 300
        self.download_file_path = ""  # Path to the downloaded file
        self.temp_working_dir = None
        self.idx_range = (None, None)


class M3Generator:
    """Class to generate M3 responses"""
    def __init__(self, model_path, conv_mode, device="cuda"):
        """Initialize the M3 generator"""
        # TODO: allow setting the device
        disable_torch_init()
        self.conv_mode = conv_mode
        self.device = device
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, self.model_name, device=self.device)
        logger.info(f"Model {self.model_name} loaded successfully. Context length: {self.context_len}")


    def generate_response(
        self,
        messages: list,
        max_tokens,
        temperature,
        top_p,
    ):
        """Generate the response"""
        logger.debug(f"Generating response with {len(messages)} messages")
        images = []

        conv = deepcopy(conv_templates[self.conv_mode])
        
        for message in messages:
            role = conv.roles[0] if message["role"] == "user" else conv.roles[1]
            prompt = ""
            for content in message["content"]:
                if content["type"] == "text":
                    prompt += content.text
                if content["type"] == "image_path":
                    images.append(load_image(content["image_path"]))
            conv.append_message(role, prompt)

        if conv.sep_style == SeparatorStyle.LLAMA_3:
            conv.append_message(conv.roles[1], "")  # add "" to the assistant message

        prompt_text = conv.get_prompt()
        logger.debug(f"Prompt input: {prompt_text}")

        if len(images) > 0:
            images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        images_input = [images_tensor] if len(images) > 0 else None

        input_ids = (
            tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

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

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        logger.debug(f"Assistant: {outputs}")

        return outputs

    def get_expert_model_args(self, model_card: dict, arg_matches: list[str]):
        """Get the expert model arguments from the text"""
        if len(arg_matches) == 0:
            return None
        if len(arg_matches) == 1 and arg_matches[0] == "":
            return None
        if len(arg_matches) > 1:
            raise ValueError("Multiple expert model arguments are provided in the same prompt"
                            "which is not supported in this version.")
        arg_match = arg_matches[0]
        if (arg_match and model_card["valid_args"] is None) or (arg_match.lower() not in model_card["valid_args"]):
            # Throw an error if
            # - the expert model does not have any valid arguments
            # - the argument is not in the valid arguments list
            raise ValueError(f"Invalid expert model argument {arg_match} for {model_card}")
        return arg_match


    def get_model_card_and_arg(self, output: str, model_cards: list):
        """Highlight the expert model messages in the text"""
        logger.debug(f"Getting expert model and arguments from the output: {output}")
        matches = re.findall(r"<(.*?)>", output)
        if len(matches) == 0:
            return None, None
        if len(matches) > 1:
            # TBD: support multiple expert models in the same prompt
            raise ValueError("Multiple expert models are called in the same prompt"
                            "which is not supported in this version.")
        match = matches[0]
        for model_card in model_cards:
            if model_card["name"].lower() in match.lower():
                arg_matches = re.findall(r"\((.*?)\)", match[len(model_card["name"]):])
                arg = self.get_expert_model_args(model_card, arg_matches)
                return model_card, arg

        # TODO: need to check if it is okay to return None, None
        raise ValueError(f"Invalid expert model: {match} is not found in the available expert models")


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


    def process_prompt(self, prompt, sv, chat_history):
        """Process the prompt and return the result. Inputs/outputs are the gradio components."""
        logger.debug(f"Process the image and return the result")

        if sv.temp_working_dir is None:
            sv.temp_working_dir = tempfile.mkdtemp()

        img_file = CACHED_IMAGES.get(sv.image_url, None)
        model_cards = [m for m in sv.model_cards if m["name"] in sv.expert_models]
        modality = get_modality(sv.image_url, text=prompt)
        sys_msg = get_system_messge(model_cards)

        if isinstance(img_file, str) and img_file.endswith(".nii.gz"):
            # Take the specific slice from a volume
            chat_history.append(prompt, image_path=get_slice_filenames(img_file, sv.slice_index)[0])
        elif isinstance(img_file, str):
            # Take the 2D image if img_file is not None
            chat_history.append(prompt, image_path=img_file)
        elif img_file is None:
            chat_history.append(prompt)
        else:
            raise ValueError(f"Invalid image file: {img_file}")

        # need squash
        

        outputs = self.generate_response(
            messages=chat_history.messages,
            max_tokens=sv.max_tokens,
            temperature=sv.temperature,
            top_p=sv.top_p,
        )

        chat_history.append(outputs, role="assistant")
        
        model_card, model_arg = self.get_model_card_and_arg(outputs, model_cards)

        if model_card:
            early_stop = check_early_stop(outputs, model_card)
            response = trigger_expert_endpoint(model_card["endpoint"], sv.image_url, model_arg)
            seg_file = None
            if model_card["task"] == "classification":
                text_output = classification_to_string(response.json())
                instruction = "Analyze the image and take these predictions into account when responding to this prompt:\n" + prompt
                chat_history.append(text_output, role="expert")
            elif model_card["task"] == "segmentation":
                seg_file = save_zipped_seg_to_file(response, sv.temp_working_dir)
                text_output = segmentation_to_string(
                    sv.temp_working_dir,
                    img_file,
                    seg_file,
                    modality=get_modality(sv.image_url, text=prompt + outputs),
                    slice_index=sv.slice_index,
                    image_filename=get_slice_filenames(img_file, sv.slice_index)[0],
                    seg_filename=get_slice_filenames(img_file, sv.slice_index)[1],
                )
                instruction = "Use this result to respond to this prompt:\n" + prompt
                chat_history.append(text_output, image_path=get_slice_filenames(img_file, sv.slice_index)[1], role="expert")

            if not early_stop:
                chat_history.append(instruction, role="assistant")
                # need squash
                outputs = self.generate_response(
                    messages=chat_history.messages,
                    max_tokens=sv.max_tokens,
                    temperature=sv.temperature,
                    top_p=sv.top_p,
                )
                chat_history.append(outputs, role="assistant")

        new_sv = SessionVariables()
        new_sv.download_file_path = seg_file if seg_file else ""
        return None, new_sv, chat_history, chat_history.get_html(show_all=False), chat_history.get_html(show_all=True)


def reset_params(sv):
    """Reset the session variables"""
    logger.debug(f"Consuming the parameters")
    if sv.download_file_path != "":
        name = os.path.basename(sv.download_file_path)
        filepath = sv.download_file_path
        sv.download_file_path = ""
        d_btn = gr.DownloadButton(label=f"Download {name}", value=filepath, visible=True)
    else:
        d_btn = gr.DownloadButton(visible=False)
    # Order of output: image, image_selector, checkboxes, slice_index_html, temperature_slider, top_p_slider, max_tokens_slider, download_button
    return sv, None, None, HARDCODED_EXPERT_MODELS, "Slice Index: N/A", 0.0, 0.9, 300, d_btn


def reset_all():
    """Clear and reset everything, Inputs/outputs are the gradio components."""
    logger.debug(f"Clearing everything")
    # Order of output: prompt_edit, chat_history, history_text, history_text_full, param_bags
    return "Enter your prompt here", ChatHistory(), HTML_PLACEHOLDER, HTML_PLACEHOLDER, SessionVariables()


def update_checkbox(checkboxes, sv):
    """Update the checkboxes"""
    logger.debug(f"Updating the checkboxes")
    sv.expert_models = checkboxes
    return sv


def update_temperature(temperature, sv):
    """Update the temperature"""
    logger.debug(f"Updating the temperature")
    sv.temperature = temperature
    return sv


def update_top_p(top_p, sv):
    """Update the top P"""
    logger.debug(f"Updating the top P")
    sv.top_p = top_p
    return sv


def update_max_tokens(max_tokens, sv):
    """Update the max tokens"""
    logger.debug(f"Updating the max tokens")
    sv.max_tokens = max_tokens
    return sv


def download_file():
    """Download the file."""
    return [gr.DownloadButton(visible=False)]



def create_demo(model_path, conv_mode):
    """Main function to create the Gradio interface"""
    generator = M3Generator(model_path, conv_mode)

    with gr.Blocks(css=CSS_STYLES) as demo:
        gr.HTML(TITLE, label="Title")
        chat_history = gr.State(value=ChatHistory())  # Prompt history
        sv = gr.State(value=SessionVariables())

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Image", placeholder="Please select an image from the dropdown list.")
                image_dropdown = gr.Dropdown(label="Select an image", choices=list(IMAGES_URLS.keys()))
                with gr.Accordion("View Parameters", open=False):
                    temperature_slider = gr.Slider(
                        label="Temperature", minimum=0.0, maximum=1.0, step=0.01, value=0.0, interactive=True
                    )
                    top_p_slider = gr.Slider(
                        label="Top P", minimum=0.0, maximum=1.0, step=0.01, value=0.9, interactive=True
                    )
                    max_tokens_slider = gr.Slider(
                        label="Max Tokens", minimum=1, maximum=1024, step=1, value=300, interactive=True
                    )

                with gr.Accordion("3D image panel", open=False):
                    slice_index_html = gr.HTML("Slice Index: N/A")
                    with gr.Row():
                        prev10_btn = gr.Button("<<")
                        prev01_btn = gr.Button("<")
                        next01_btn = gr.Button(">")
                        next10_btn = gr.Button(">>")

            with gr.Column():
                with gr.Tab("In front of the scene"):
                    history_text = gr.HTML(HTML_PLACEHOLDER, label="Previous prompts")
                with gr.Tab("Behind the scene"):
                    history_text_full = gr.HTML(HTML_PLACEHOLDER, label="Previous prompts full")
                image_download = gr.DownloadButton("Download the file", visible=False)
                clear_btn = gr.Button("Clear Conversation")
                with gr.Row(variant="compact"):
                    prompt_edit = gr.Textbox(label="Enter your prompt here", container=False, placeholder="Enter your prompt here", scale=2)
                    submit_btn = gr.Button("Submit", scale=0)
                gr.Examples(EXAMPLE_PROMPTS, prompt_edit)
                checkboxes = gr.CheckboxGroup(
                    choices=[m["name"] for m in MODEL_CARDS],
                    value=[m["name"] for m in MODEL_CARDS],
                    label="Expert Models",
                    info="Select the expert models to use.",
                )

        # Process image and clear it immediately by returning None
        submit_btn.click(
            fn=generator.process_prompt,
            inputs=[prompt_edit, sv, chat_history],
            outputs=[prompt_edit, sv, chat_history, history_text, history_text_full],
        )
        prompt_edit.submit(
            fn=generator.process_prompt,
            inputs=[prompt_edit, sv, chat_history],
            outputs=[prompt_edit, sv, chat_history, history_text, history_text_full],
        )

        # Param controlling buttons
        image_input.input(fn=input_image, inputs=[image_input, sv], outputs=[image_input, sv])
        image_dropdown.change(
            fn=update_image_selection,
            inputs=[image_dropdown, sv, slice_index_html],
            outputs=[image_input, sv, slice_index_html],
        )
        prev10_btn.click(
            fn=update_image_prev_10,
            inputs=[image_dropdown, sv, slice_index_html],
            outputs=[image_input, sv, slice_index_html],
        )
        prev01_btn.click(
            fn=update_image_prev_1,
            inputs=[image_dropdown, sv, slice_index_html],
            outputs=[image_input, sv, slice_index_html],
        )
        next01_btn.click(
            fn=update_image_next_1,
            inputs=[image_dropdown, sv, slice_index_html],
            outputs=[image_input, sv, slice_index_html],
        )
        next10_btn.click(
            fn=update_image_next_10,
            inputs=[image_dropdown, sv, slice_index_html],
            outputs=[image_input, sv, slice_index_html],
        )
        checkboxes.change(fn=update_checkbox, inputs=[checkboxes, sv], outputs=[sv])
        temperature_slider.change(fn=update_temperature, inputs=[temperature_slider, sv], outputs=[sv])
        top_p_slider.change(fn=update_top_p, inputs=[top_p_slider, sv], outputs=[sv])
        max_tokens_slider.change(fn=update_max_tokens, inputs=[max_tokens_slider, sv], outputs=[sv])

        # Reset button
        clear_btn.click(
            fn=reset_all, inputs=[], outputs=[prompt_edit, chat_history, history_text, history_text_full, sv]
        )

        # States
        sv.change(
            fn=reset_params,
            inputs=[sv],
            outputs=[
                sv,
                image_input,
                image_dropdown,
                checkboxes,
                slice_index_html,
                temperature_slider,
                top_p_slider,
                max_tokens_slider,
                image_download,
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Add the argument to load multiple models from a JSON file
    parser.add_argument("--conv_mode", type=str, default="VILA2D", help="The conversation mode to use.")
    parser.add_argument("--model_path", type=str, default="VILA2D", help="The path to the model to load.")
    args = parser.parse_args()
    # Save the config to a file
    with open("config.json", "w") as f:
        json.dump(vars(args), f)
    cache_images()
    demo = create_demo(args.model_path, args.conv_mode)
    demo.launch(server_name="0.0.0.0", server_port=7860)
    cache_cleanup()
