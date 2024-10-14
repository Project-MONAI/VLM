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
import shutil
import tempfile
import urllib.request
from io import BytesIO
from zipfile import ZipFile

import gradio as gr
import nibabel as nib
import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI
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


def get_models(nvcf=False):
    """Get the models"""
    if nvcf:
        # Fixed models for the NVCF API
        return {
            "nvcf-8b": {
                "checkpoint": "tumor_expert_alldata_4node_model_8bfix_aug_29_2024_run2_e3.0/checkpoint-3500",
                "url": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/319c3e8e-5913-4577-8223-a7484766f41a",
                "type": "requests",
            },
        }
    
    return {
        "baseline-server-8b": {
            "url": "http://dlmed-api-m3.nvidia.com:8000",
            "type": "openai",
        },
        "baseline-server-13b": {
            "url": "http://dlmed-api-m3.nvidia.com:8001",
            "type": "openai",
        },
        "baseline-server-alldata-13b": {
            "url": "http://dlmed-api-m3.nvidia.com:8002",
            "type": "openai",
        },
    }


def cleanup_cache():
    """Clean up the cache"""
    logger.debug(f"Cleaning up the cache")
    for _, cache_file_name in CACHED_IMAGES.items():
        if os.path.exists(cache_file_name):
            os.remove(cache_file_name)
            print(f"Cache file {cache_file_name} cleaned up")


def load_nii_to_numpy(nii_path, slice_index=None, axis=2):
    """
    Load a .nii.gz file and convert it to a Pillow RGB image.

    Parameters:
    - nii_path: str, URL to the .nii.gz file
    - slice_index: int, the index of the slice to extract (default is 0)
    - axis: int, the axis along which to take the slice (default is 2 for axial slices)

    Returns:
    - img: Pillow Image in RGB format
    """
    logger.debug(f"Loading NII file to NumPy array")
    # Get the image data as a NumPy array
    nii = nib.load(nii_path)
    data = nii.get_fdata()

    # Select a 2D slice from the 3D volume along the specified axis
    slice_index = data.shape[axis] // 2 if slice_index is None else slice_index
    slice_index = min(slice_index, data.shape[axis] - 1)
    slice_index = max(slice_index, 0)
    slice_data = np.take(data, slice_index, axis=axis)

    # Normalize the slice to the range [0, 255] for image display
    slice_data = np.clip(slice_data, np.min(slice_data), np.max(slice_data))
    slice_data = ((slice_data - np.min(slice_data)) / (np.ptp(slice_data))) * 255
    # rotate the image
    slice_data = np.rot90(slice_data)
    return slice_data.astype(np.uint8), slice_index


def unzip_file_contents(file_content):
    """Unzip the file contents and return the image and segmentation file paths"""
    zip_file = BytesIO(base64.b64decode(file_content))
    temp_folder = tempfile.mkdtemp()
    zip_file_path = CACHED_DIR + "/output.zip"
    with open(zip_file_path, "wb") as f:
        f.write(zip_file.getvalue())

    with ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(temp_folder)

    os.remove(zip_file_path)

    print(f"Extracted files: {os.listdir(temp_folder)}")
    image_files = []
    mask_file = None
    for file in os.listdir(temp_folder):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            image_files.append(file)
        if file.endswith(".nrrd"):
            # remove the file if it exists
            mask_file = os.path.join(CACHED_DIR, file)
            if os.path.exists(mask_file):
                os.remove(mask_file)
            shutil.move(os.path.join(temp_folder, file), CACHED_DIR)

    img_file = None
    seg_file = None
    for file in image_files:
        if "image" in file:
            shutil.move(os.path.join(temp_folder, file), CACHED_DIR)
            img_file = os.path.join(CACHED_DIR, file)
        elif "label" in file:
            shutil.move(os.path.join(temp_folder, file), CACHED_DIR)
            seg_file = os.path.join(CACHED_DIR, file)
    return img_file, seg_file, mask_file


def input_image(image, params_bag):
    """Update the params bag with the input image data URL if it's inputted by the user"""
    logger.debug(f"Received user input image")
    params_bag.image_url = image_to_data_url(image)
    return image, params_bag


def update_image(selected_image, params_bag, slice_index_html, increment=None):
    """Update the gradio components based on the selected image"""
    logger.debug(f"Updating display image for {selected_image}")
    params_bag.image_url = IMAGES_URLS.get(selected_image, None)

    if params_bag.image_url is None:
        return None, params_bag, slice_index_html

    if params_bag.image_url.endswith(".nii.gz"):
        logger.debug(f"Downloading 3D sample images to {CACHED_DIR}")
        if increment is not None and params_bag.slice_index is not None:
            params_bag.slice_index += increment

        local_path = CACHED_IMAGES.get(params_bag.image_url, None)
        if local_path is None:
            basename = os.path.basename(params_bag.image_url)
            CACHED_IMAGES[params_bag.image_url] = local_path = os.path.join(CACHED_DIR, basename)
            with urllib.request.urlopen(params_bag.image_url) as response:
                with open(local_path, "wb") as f:
                    f.write(response.read())

        image, params_bag.slice_index = load_nii_to_numpy(local_path, slice_index=params_bag.slice_index)
        # This `image` will not be used to display. We don't need to convert it to data URL
        return image, params_bag, f"Slice Index: {params_bag.slice_index}"

    params_bag.slice_index = None
    return (
        params_bag.image_url,
        params_bag,
        "Slice Index: N/A for 2D images, clicking prev/next will not change the image.",
    )


def update_image_next_10(selected_image, params_bag, slice_index_html):
    """Update the image to the next 10 slices"""
    return update_image(selected_image, params_bag, slice_index_html, increment=10)


def update_image_next_1(selected_image, params_bag, slice_index_html):
    """Update the image to the next slice"""
    return update_image(selected_image, params_bag, slice_index_html, increment=1)


def update_image_prev_1(selected_image, params_bag, slice_index_html):
    """Update the image to the previous slice"""
    return update_image(selected_image, params_bag, slice_index_html, increment=-1)


def update_image_prev_10(selected_image, params_bag, slice_index_html):
    """Update the image to the previous 10 slices"""
    return update_image(selected_image, params_bag, slice_index_html, increment=-10)


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
                    "type": "image_url",
                    "image_url": image_url
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

    def append(self, prompt_or_answer, image_url=None, slice_index=None, role="user"):
        """
        Append a new message to the chat history.

        Args:
            prompt_or_answer (str): The text prompt from human or answer from AI to append.
            img (np.Array): The image to append. Optional. Only used for debugging.
            image_url (str): The image URL to append.
            role (str): The role of the message. Default is "user". Other option is "assistant".
        """
        new_contents = [
            {
                "type": "text",
                "text": prompt_or_answer,
            }
        ]
        if image_url is not None:
            new_contents.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "slice_index": slice_index,
                    },
                }
            )

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
                elif content["type"] == "image_url":
                    image_url = content["image_url"]["url"]
                    # Convert the image URL to a data URL
                    if not image_url.startswith("data:image"):
                        data_url = image_to_data_url(image_url, max_size=(300, 300))
                    else:
                        data_url = resize_data_url(image_url, (300, 300))
                    history_text_html += colorcode_message(
                        data_url=data_url, show_all=True, role=role
                    )  # always show the image
                else:
                    raise ValueError(f"Invalid content type: {content['type']}")
            history.append(history_text_html)
        return "<br>".join(history)

    def replace_last(self, image_url, role="user"):
        """Replace the last message in the chat history"""
        logger.debug(f"Replacing the last message in the chat history")
        if len(self.messages) == 0:
            return
        for message in reversed(self.messages):
            if message["role"] == role:
                for content in reversed(self.messages[-1]["content"]):
                    if content["type"] == "image_url":
                        content["image_url"]["url"] = image_url
                    return


class ParamsBag:
    """Class to store the parameters"""

    expert_models = HARDCODED_EXPERT_MODELS  # Expert models to use
    slice_index = None  # Slice index for 3D images
    image_url = None  # Image URL to display and process
    top_p = 0.9 
    temperature = 0.0
    max_tokens = 300
    download_file_path = ""  # Path to the downloaded file
    model_index = 0  # Index of the selected model among a list of models to try. Useful under unrestricted mode.


class ApiAdapter:
    """Adapter class to interact with the OpenAI API or the NVCF API with requests"""

    def __init__(self, base_url, type="openai", api_key="fake-key"):
        """Initialize the API adapter"""
        self.type = type
        self.base_url = base_url
        if type == "openai":
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        elif type == "requests":
            self.client = requests.session()
            self.client.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "accept": "application/json",
            }
        else:
            raise ValueError(f"Invalid API type: {type}")
        self.response = None

    def _chat_openai(
        self, messages=[], max_tokens=300, temperature=0.0, top_p=0.9, model="M3", exclude_model_cards=[], stream=False
    ):
        """Chat with the OpenAI API"""
        self.response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            model=model,
            extra_body={"exclude_model_cards": exclude_model_cards},
            stream=stream,
        )

    def _chat_requests(
        self, messages=[], max_tokens=300, temperature=0.0, top_p=0.9, model="M3", exclude_model_cards=[], stream=False
    ):
        """Chat with the NVCF API using requests"""
        response = self.client.post(
            self.base_url,
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "model": model,
                "exclude_model_cards": exclude_model_cards,
                "stream": stream,
            },
        )
        response.raise_for_status()
        self.response = response.json()

    def chat_completion(self, **kwargs):
        """Chat with the API"""
        return self._chat_openai(**kwargs) if self.type == "openai" else self._chat_requests(**kwargs)

    def get_file_content(self):
        """Get the file content from the response"""
        return self.response.file_content if self.type == "openai" else self.response["file_content"]

    def get_choices(self):
        """Get the choices from the response"""
        return self.response.choices if self.type == "openai" else self.response["choices"]

    def get_role(self, choice):
        """Get the role of the choice"""
        return choice.message.role if self.type == "openai" else choice["message"]["role"]

    def get_content(self, choice):
        """Get the content of the choice"""
        return choice.message.content if self.type == "openai" else choice["message"]["content"]


def reset_params(params_bag):
    """Operate UI components and state after the parameters from the params bag are consumed"""
    logger.debug(f"Consuming the parameters")
    # Order of output: image, image_selector, checkboxes, slice_index_html, temperature_slider, top_p_slider, max_tokens_slider
    if params_bag.download_file_path != "":
        name = os.path.basename(params_bag.download_file_path)
        filepath = params_bag.download_file_path
        params_bag.download_file_path = ""
        d_btn = gr.DownloadButton(label=f"Download {name}", value=filepath, visible=True)
    else:
        d_btn = gr.DownloadButton(visible=False)
    models = get_models()
    model_keys = list(models.keys())
    model_choice = model_keys[params_bag.model_index]
    return params_bag, None, None, HARDCODED_EXPERT_MODELS, "Slice Index: N/A", 0.0, 0.9, 300, d_btn, model_choice


def process_prompt(prompt, params_bag, chat_history):
    """Process the prompt and return the result. Inputs/outputs are the gradio components."""
    logger.debug(f"Process the image and return the result")

    chat_history.append(prompt, image_url=params_bag.image_url, slice_index=params_bag.slice_index)
    model_index = params_bag.model_index  # Keep the model selection for the next round
    models = get_models()
    model_choice = list(models.keys())[model_index]
    base_url = models[model_choice]["url"]
    api_type = models[model_choice]["type"]
    api_adaptor = ApiAdapter(base_url, type=api_type, api_key=os.getenv("api_key", "fake-key"))

    exclude_model_cards = []
    for model in HARDCODED_EXPERT_MODELS:
        if model not in params_bag.expert_models:
            exclude_model_cards.append(model)
    logger.debug(f"Excluding model cards: {exclude_model_cards}")
    api_adaptor.chat_completion(
        messages=chat_history.messages,
        max_tokens=params_bag.max_tokens,
        temperature=params_bag.temperature,
        top_p=params_bag.top_p,
        model="M3",
        exclude_model_cards=exclude_model_cards,
        stream=False,
    )

    img_file, seg_file, mask_file = unzip_file_contents(api_adaptor.get_file_content())
    if img_file is not None:
        chat_history.replace_last(image_to_data_url(img_file))
        os.remove(img_file)
    for choice in api_adaptor.get_choices():
        role = api_adaptor.get_role(choice)
        content = api_adaptor.get_content(choice)
        if role == "expert" and seg_file is not None and "<segmentation>" in content:
            logger.debug(f"Segmentation image found in the expert response")
            seg_url = image_to_data_url(seg_file)
            os.remove(seg_file)
            chat_history.append(content, role=role, image_url=seg_url)
        else:
            logger.debug(f"Appending the response to the chat history")
            chat_history.append(content, role=role)

    new_params_bag = ParamsBag()
    new_params_bag.download_file_path = mask_file if mask_file else ""
    new_params_bag.model_index = model_index  # Keep the model
    return None, new_params_bag, chat_history, chat_history.get_html(show_all=False), chat_history.get_html(show_all=True)


def clear_label():
    """Clear and reset everything, Inputs/outputs are the gradio components."""
    logger.debug(f"Clearing everything")
    # Order of output: prompt_edit, chat_history, history_text, history_text_full, param_bags
    return "Enter your prompt here", ChatHistory(), HTML_PLACEHOLDER, HTML_PLACEHOLDER, ParamsBag()


def update_checkbox(checkboxes, params_bag):
    """Update the checkboxes"""
    logger.debug(f"Updating the checkboxes")
    params_bag.expert_models = checkboxes
    return params_bag


def update_temperature(temperature, params_bag):
    """Update the temperature"""
    logger.debug(f"Updating the temperature")
    params_bag.temperature = temperature
    return params_bag


def update_top_p(top_p, params_bag):
    """Update the top P"""
    logger.debug(f"Updating the top P")
    params_bag.top_p = top_p
    return params_bag


def update_max_tokens(max_tokens, params_bag):
    """Update the max tokens"""
    logger.debug(f"Updating the max tokens")
    params_bag.max_tokens = max_tokens
    return params_bag


def download_file():
    """Download the file."""
    return [gr.DownloadButton(visible=False)]


def update_checkpoint(selected_model_index, params_bag):
    """Update the checkpoint"""
    logger.debug(f"Updating the checkpoint with {selected_model_index}")
    params_bag.model_index = selected_model_index
    return params_bag


def main(args):
    """Main function to create the Gradio interface"""
    def generate_css():
        """Generate CSS"""
        css = ".fixed-size-image {\n"
        css += "width: 512px;\n"
        css += "height: 512px;\n"
        css += "object-fit: cover;\n"
        css += "}\n"
        css += ".small-text {\n"
        css += "font-size: 6px;\n"
        css += "}\n"
        return css

    with gr.Blocks(css=generate_css()) as demo:
        is_debug = not args.restricted
        logger.debug(f"Running the demo with debug mode: {is_debug}")
        gr.HTML(TITLE, label="Title")
        chat_history = gr.State(value=ChatHistory())  # Prompt history
        params_bag = gr.State(value=ParamsBag())

        with gr.Row():
            with gr.Column():
                image_sources = ["upload", "webcam", "clipboard"] if is_debug else []
                image_input = gr.Image(
                    label="Image",
                    sources=image_sources,
                    placeholder="Please select an 2D or 3D slice from the dropdown list.",
                )
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
                models = get_models(args.nvcf)
                model_keys = list(models.keys())
                model_dropdown = gr.Dropdown(
                    label="Select a model", choices=list(models.keys()), value=model_keys[0], type="index", visible=is_debug
                )
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
                    choices=HARDCODED_EXPERT_MODELS,
                    value=HARDCODED_EXPERT_MODELS,
                    label="Expert Models",
                    info="Select the expert models to use.",
                )

        # Process image and clear it immediately by returning None
        submit_btn.click(
            fn=process_prompt,
            inputs=[prompt_edit, params_bag, chat_history],
            outputs=[prompt_edit, params_bag, chat_history, history_text, history_text_full],
        )
        prompt_edit.submit(
            fn=process_prompt,
            inputs=[prompt_edit, params_bag, chat_history],
            outputs=[prompt_edit, params_bag, chat_history, history_text, history_text_full],
        )

        # Param controlling buttons
        image_input.input(fn=input_image, inputs=[image_input, params_bag], outputs=[image_input, params_bag])
        image_dropdown.change(
            fn=update_image,
            inputs=[image_dropdown, params_bag, slice_index_html],
            outputs=[image_input, params_bag, slice_index_html],
        )
        prev10_btn.click(
            fn=update_image_prev_10,
            inputs=[image_dropdown, params_bag, slice_index_html],
            outputs=[image_input, params_bag, slice_index_html],
        )
        prev01_btn.click(
            fn=update_image_prev_1,
            inputs=[image_dropdown, params_bag, slice_index_html],
            outputs=[image_input, params_bag, slice_index_html],
        )
        next01_btn.click(
            fn=update_image_next_1,
            inputs=[image_dropdown, params_bag, slice_index_html],
            outputs=[image_input, params_bag, slice_index_html],
        )
        next10_btn.click(
            fn=update_image_next_10,
            inputs=[image_dropdown, params_bag, slice_index_html],
            outputs=[image_input, params_bag, slice_index_html],
        )
        checkboxes.change(fn=update_checkbox, inputs=[checkboxes, params_bag], outputs=[params_bag])
        temperature_slider.change(fn=update_temperature, inputs=[temperature_slider, params_bag], outputs=[params_bag])
        top_p_slider.change(fn=update_top_p, inputs=[top_p_slider, params_bag], outputs=[params_bag])
        max_tokens_slider.change(fn=update_max_tokens, inputs=[max_tokens_slider, params_bag], outputs=[params_bag])
        model_dropdown.change(fn=update_checkpoint, inputs=[model_dropdown, params_bag], outputs=[params_bag])

        # Reset button
        clear_btn.click(
            fn=clear_label, inputs=[], outputs=[prompt_edit, chat_history, history_text, history_text_full, params_bag]
        )

        # States
        params_bag.change(
            fn=reset_params,
            inputs=[params_bag],
            outputs=[
                params_bag,
                image_input,
                image_dropdown,
                checkboxes,
                slice_index_html,
                temperature_slider,
                top_p_slider,
                max_tokens_slider,
                image_download,
                model_dropdown,
            ],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
    cleanup_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restricted", action="store_true", help="Run the demo in restricted mode")
    parser.add_argument("--nvcf", action="store_true", help="Use the NVCF API")
    args = parser.parse_args()

    main(args)
