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
import html
import logging
import os
import tempfile
import time 
from copy import deepcopy
from glob import glob
from shutil import copyfile, rmtree

import gradio as gr
import nibabel as nib
import torch
from dotenv import load_dotenv
from experts.expert_monai_vista3d import ExpertVista3D
from experts.expert_torchxrayvision import ExpertTXRV
from experts.utils import (
    get_modality,
    get_monai_transforms,
    get_slice_filenames,
    image_to_data_url,
    load_image,
    save_image_url_to_file,
)
from huggingface_hub import snapshot_download
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from tqdm import tqdm

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

# Sample images dictionary. It accepts either a URL or a local path.
IMG_URLS_OR_PATHS = {
    # "CT Sample 1": "https://developer.download.nvidia.com/assets/Clara/monai/samples/liver_0.nii.gz",
    # "CT Sample 2": "https://developer.download.nvidia.com/assets/Clara/monai/samples/ct_sample.nii.gz",
    "Chest X-ray Sample 1": "https://developer.download.nvidia.com/assets/Clara/monai/samples/cxr_8e067d88-2ea4ee8d-21db2c6b-f78701cb-91ad53f9_v1.jpg",
    "Chest X-ray Sample 2": "https://developer.download.nvidia.com/assets/Clara/monai/samples/cxr_51e9421b-c2f395da-5dd48889-7e307aca-1472d6a6_v1.jpg",
    "Chest X-ray Sample 3": "https://developer.download.nvidia.com/assets/Clara/monai/samples/cxr_c2af2ab3-6a11cbae-d9fa4d64-21ab221e-cf6f2146_v1.jpg",
}

MODEL_CARDS = "Here is a list of available expert models:\n<BRATS(args)> Modality: MRI, Task: segmentation, Overview: A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data, Accuracy: Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 - Average: 0.8518, Valid args are: None\n<VISTA3D(args)> Modality: CT, Task: segmentation, Overview: domain-specialized interactive foundation model developed for segmenting and annotating human anatomies with precision, Accuracy: 127 organs: 0.792 Dice on average, Valid args are: 'everything', 'hepatic tumor', 'pancreatic tumor', 'lung tumor', 'bone lesion', 'organs', 'cardiovascular', 'gastrointestinal', 'skeleton', or 'muscles'\n<VISTA2D(args)> Modality: cell imaging, Task: segmentation, Overview: model for cell segmentation, which was trained on a variety of cell imaging outputs, including brightfield, phase-contrast, fluorescence, confocal, or electron microscopy, Accuracy: Good accuracy across several cell imaging datasets, Valid args are: None\n<CXR(args)> Modality: chest x-ray (CXR), Task: classification, Overview: pre-trained model which are trained on large cohorts of data, Accuracy: Good accuracy across several diverse chest x-rays datasets, Valid args are: None\nGive the model <NAME(args)> when selecting a suitable expert model.\n"

SYS_PROMPT = None  # set when the script initializes

EXAMPLE_PROMPTS_3D = [
    ["Segment the visceral structures in the current image."],
    ["Can you identify any liver masses or tumors?"],
    ["Segment the entire image."],
    ["What's in the scan?"],
    ["Segment the muscular structures in this image."],
    ["Could you please isolate the cardiovascular system in this image?"],
    ["Separate the gastrointestinal region from the surrounding tissue in this image."],
    ["Can you assist me in segmenting the bony structures in this image?"],
    ["Describe the image in detail"],
]

EXAMPLE_PROMPTS_2D = [
    ["What abnormalities are seen in this image?"],
    ["Is there evidence of edema in this image?"],
    ["Is there pneumothorax?"],
    ["What type is the lung opacity?"],
    ["Which view is this image taken?"],
    ["Is there evidence of cardiomegaly in this image?"],
    ["Is the atelectasis located on the left side or right side?"],
    ["What level is the cardiomegaly?"],
    ["Describe the image in detail"],
]

HTML_PLACEHOLDER = "<br>".join([""] * 15)

CACHED_DIR = tempfile.mkdtemp()

CACHED_IMAGES = {}

TITLE = """
<div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px;">
    <p>
        <img src="https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/MONAI-logo-color.png" alt="Project MONAI logo" 
            style="width: 50%; max-width: 600px; min-width: 300px; margin: auto; display: block;">
    </p>
    
    <h1 style="font-weight: 900; font-size: 1.5rem; margin-bottom: 10px;">
        MONAI Multi-modal Model (M3) VLM Demo
    </h1>
    
    <div style="font-size: 0.95rem; text-align: left; max-width: 800px; margin: 0 auto;">
        <span>
            VILA-M3 is a vision-language model for medical applications that interprets medical images and text prompts to generate relevant responses.
        </span>
        <details style="display: inline; cursor: pointer;">
            <summary style="display: inline; font-weight: bold;">
                <strong>DISCLAIMER</strong>
            </summary>
            <span style="font-size: 0.95rem;">
                AI models generate responses and outputs based on complex algorithms and machine learning techniques, 
                and those responses or outputs may be inaccurate, harmful, biased, or indecent. By testing this model, you assume the risk of any 
                harm caused by any response or output of the model. This model is for research purposes and not for clinical usage.
            </span>
        </details>
    </div>
</div>
"""

CSS_STYLES = (
    ".fixed-size-image {\n"
    "width: 512px;\n"
    "height: 512px;\n"
    "object-fit: cover;\n"
    "}\n"
    ".small-text {\n"
    "font-size: 6px;\n"
    "}\n"
)


def cache_images():
    """Cache the image and return the file path"""
    logger.debug(f"Caching the image to {CACHED_DIR}")
    for _, item in IMG_URLS_OR_PATHS.items():
        if item.startswith("http"):
            CACHED_IMAGES[item] = save_image_url_to_file(item, CACHED_DIR)
        elif os.path.exists(item):
            # move the file to the cache directory
            file_name = os.path.basename(item)
            CACHED_IMAGES[item] = os.path.join(CACHED_DIR, file_name)
            if not os.path.isfile(CACHED_IMAGES[item]):
                copyfile(item, CACHED_IMAGES[item])

        if CACHED_IMAGES[item].endswith(".nii.gz"):
            data = nib.load(CACHED_IMAGES[item]).get_fdata()
            for slice_index in tqdm(range(data.shape[2])):
                image_filename = get_slice_filenames(CACHED_IMAGES[item], slice_index)[0]
                if not os.path.exists(os.path.join(CACHED_DIR, image_filename)):
                    compose = get_monai_transforms(
                        ["image"],
                        CACHED_DIR,
                        modality="CT",  # TODO: Get the modality from the image/prompt/metadata
                        slice_index=slice_index,
                        image_filename=image_filename,
                    )
                    compose({"image": CACHED_IMAGES[item]})


def cache_cleanup():
    """Clean up the cache"""
    logger.debug(f"Cleaning up the cache")
    rmtree(CACHED_DIR)


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

    def get_html(self, show_all: bool = False, sys_msgs_to_hide: list | None = None):
        """Returns the chat history as an HTML string to display"""
        history = []

        for message in self.messages:
            role = message["role"]
            contents = message["content"]
            history_text_html = ""
            for content in contents:
                if content["type"] == "text":
                    history_text_html += colorcode_message(
                        text=content["text"], show_all=show_all, role=role, sys_msgs_to_hide=sys_msgs_to_hide
                    )
                else:
                    history_text_html += colorcode_message(
                        data_url=image_to_data_url(content["image_path"], max_size=(300, 300)), show_all=True, role=role
                    )  # always show the image
            history.append(history_text_html)
        return "<br>".join(history)


class SessionVariables:
    """Class to store the session variables"""

    def __init__(self):
        """Initialize the session variables"""
        self.sys_prompt = SYS_PROMPT
        self.sys_msg = MODEL_CARDS
        self.use_model_cards = False
        self.slice_index = None  # Slice index for 3D images
        self.image_url = None  # Image URL to the image on the web
        self.axis = 2
        self.top_p = 0.9
        self.temperature = 0.0
        self.max_tokens = 1024
        self.download_file_path = ""  # Path to the downloaded file
        self.temp_working_dir = None
        self.idx_range = (None, None)
        self.interactive = False
        self.sys_msgs_to_hide = []
        self.modality_prompt = "Unknown"


def new_session_variables(**kwargs):
    """Create a new session variables but keep the conversation mode"""
    if len(kwargs) == 0:
        return SessionVariables()
    sv = SessionVariables()
    for key, value in kwargs.items():
        if sv.__getattribute__(key) != value:
            sv.__setattr__(key, value)
    return sv


class M3Generator:
    """Class to generate M3 responses"""

    def __init__(self, source="huggingface", model_path="", conv_mode="", device="cuda"):
        """Initialize the M3 generator"""
        global SYS_PROMPT
        self.source = source
        # TODO: support huggingface models
        if source == "local":
            # TODO: allow setting the device
            disable_torch_init()
            self.conv_mode = conv_mode
            model_name = get_model_name_from_path(model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path, model_name
            )
            logger.info(f"Model {model_name} loaded successfully. Context length: {self.context_len}")
        elif source == "huggingface":
            model_path = snapshot_download("MONAI/Llama3-VILA-M3-13B")
            self.conv_mode = "vicuna_v1"
            model_name = get_model_name_from_path(model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path, model_name
            )
            logger.info(f"Model {model_name} loaded successfully. Context length: {self.context_len}")
        else:
            raise NotImplementedError(f"Source {source} is not supported.")
        SYS_PROMPT = conv_templates[self.conv_mode].system  # only set once

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
                    images.append(load_image(content["image_path"]))
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

        input_ids = (
            tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )

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

    def generate_response(self, **kwargs):
        """Generate the response"""
        if self.source == "local" or self.source == "huggingface":
            return self.generate_response_local(**kwargs)
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

            i += 1  # Move to the next message. TODO: Check if this is correct

        return messages

    def process_prompt(self, prompt, sv, chat_history):
        """Process the prompt and return the result. Inputs/outputs are the gradio components."""
        logger.debug(f"Process the image and return the result")

        if not sv.interactive:
            # Do not process the prompt if the image is not provided
            return None, sv, chat_history, "Please select an image", "Please select an image"

        if sv.temp_working_dir is None:
            sv.temp_working_dir = tempfile.mkdtemp()

        if sv.modality_prompt == "Auto":
            modality = get_modality(sv.image_url, text=prompt)
        else:
            modality = sv.modality_prompt
        mod_msg = f"This is a {modality} image.\n" if modality != "Unknown" else ""

        model_cards = sv.sys_msg if sv.use_model_cards else ""

        img_file = CACHED_IMAGES.get(sv.image_url, None)

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
                    image_path=os.path.join(CACHED_DIR, get_slice_filenames(img_file, sv.slice_index)[0]),
                )
            else:
                chat_history.append(_prompt, image_path=img_file)
        elif img_file is None:
            # text-only prompt
            chat_history.append(prompt)  # no image token
        else:
            raise ValueError(f"Invalid image file: {img_file}")

        # need squash
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
        download_pkg = ""

        for expert_model in [ExpertTXRV, ExpertVista3D]:
            expert = expert_model() if expert_model().mentioned_by(outputs) else None
            if expert:
                break

        if expert:
            logger.debug(f"Expert model {expert.__class__.__name__} is being called to process {sv.image_url}.")
            try:
                if sv.image_url is None:
                    raise ValueError(f"No image is provided with {outputs}.")
                text_output, seg_file, instruction, download_pkg = expert.run(
                    image_url=sv.image_url,
                    input=outputs,
                    output_dir=sv.temp_working_dir,
                    img_file=img_file,
                    slice_index=sv.slice_index,
                    prompt=prompt,
                )
            except Exception as e:
                text_output = f"Sorry I met an error: {e}"
                seg_file = None
                instruction = ""
                download_pkg = ""

            chat_history.append(text_output, image_path=seg_file, role="expert")
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

        new_sv = new_session_variables(
            # Keep these parameters accross one conversation
            sys_prompt=sv.sys_prompt,
            sys_msg=sv.sys_msg,
            use_model_cards=sv.use_model_cards,
            download_file_path=download_pkg,
            temp_working_dir=sv.temp_working_dir,
            interactive=True,
            sys_msgs_to_hide=sv.sys_msgs_to_hide,
        )
        return (
            None,
            new_sv,
            chat_history,
            chat_history.get_html(show_all=False, sys_msgs_to_hide=sv.sys_msgs_to_hide),
            chat_history.get_html(show_all=True),
        )


def input_image(image, sv: SessionVariables):
    """Update the session variables with the input image data URL if it's inputted by the user"""
    logger.debug(f"Received user input image")
    # TODO: support user uploaded images
    return image, sv


def update_image_selection(selected_image, sv: SessionVariables, slice_index=None):
    """Update the gradio components based on the selected image"""
    logger.debug(f"Updating display image for {selected_image}")
    sv.image_url = IMG_URLS_OR_PATHS.get(selected_image, None)
    img_file = CACHED_IMAGES.get(sv.image_url, None)

    if sv.image_url is None or img_file is None:
        return None, sv, gr.Slider(0, 2, 1, 0, visible=False), [[""]]

    sv.interactive = True
    if img_file.endswith(".nii.gz"):
        if slice_index is None:
            slice_file_pttn = img_file.replace(".nii.gz", "*_img.jpg")
            # glob the image files
            slice_files = glob(slice_file_pttn)
            sv.slice_index = len(slice_files) // 2
            sv.idx_range = (0, len(slice_files) - 1)
        else:
            # Slice index is updated by the slidebar.
            # There is no need to update the idx_range.
            sv.slice_index = slice_index

        image_filename = get_slice_filenames(img_file, sv.slice_index)[0]
        if not os.path.exists(os.path.join(CACHED_DIR, image_filename)):
            raise ValueError(f"Image file {image_filename} does not exist.")
        return (
            os.path.join(CACHED_DIR, image_filename),
            sv,
            gr.Slider(sv.idx_range[0], sv.idx_range[1], value=sv.slice_index, step=1, visible=True, interactive=True),
            gr.Dataset(samples=EXAMPLE_PROMPTS_3D),
        )

    sv.slice_index = None
    sv.idx_range = (None, None)
    return (
        img_file,
        sv,
        gr.Slider(0, 2, 1, 0, visible=False),
        gr.Dataset(samples=EXAMPLE_PROMPTS_2D),
    )


def colorcode_message(text="", data_url=None, show_all=False, role="user", sys_msgs_to_hide: list = None):
    """Color the text based on the role and return the HTML text"""
    logger.debug(f"Preparing the HTML text with {show_all} and role: {role}")
    # if content is not a data URL, escape the text

    if not show_all and role == "expert":
        return ""
    if not show_all and sys_msgs_to_hide and isinstance(sys_msgs_to_hide, list):
        sys_msgs_to_hide.sort(key=len, reverse=True)
        for sys_msg in sys_msgs_to_hide:
            text = text.replace(sys_msg, "")

    escaped_text = html.escape(text)
    # replace newlines with <br> tags
    escaped_text = escaped_text.replace("\n", "<br>")
    if data_url is not None:
        escaped_text += f'<img src="{data_url}">'
    if role == "user":
        return f'<p style="color: blue;">User:</p> {escaped_text}'
    elif role == "expert":
        return f'<p style="color: green;">Expert:</p> {escaped_text}'
    elif role == "assistant":
        return f'<p style="color: red;">AI Assistant:</p> {escaped_text}</p>'
    raise ValueError(f"Invalid role: {role}")


def clear_one_conv(sv: SessionVariables):
    """
    Post-event hook indicating the session ended.It's called when `new_session_variables` finishes.
    Particularly, it resets the non-text parameters. So it excludes:
        - prompt_edit
        - chat_history
        - history_text
        - history_text_full
        - sys_prompt_text
        - model_cards_text
    If some of the parameters need to stay persistent in the session, they should be modified in the `clear_all_convs` function.
    """
    logger.debug(f"Clearing the parameters of one conversation")
    if sv.download_file_path != "":
        name = os.path.basename(sv.download_file_path)
        filepath = sv.download_file_path
        sv.download_file_path = ""
        d_btn = gr.DownloadButton(label=f"Download {name}", value=filepath, visible=True)
    else:
        d_btn = gr.DownloadButton(visible=False)
    # Order of output: image, image_selector, temperature_slider, top_p_slider, max_tokens_slider, download_button, image_slider
    return sv, None, None, sv.temperature, sv.top_p, sv.max_tokens, d_btn, gr.Slider(0, 2, 1, 0, visible=False)


def clear_all_convs(sv: SessionVariables):
    """Clear and reset everything, Inputs/outputs are the gradio components."""
    logger.debug(f"Clearing all conversations")
    if sv.temp_working_dir is not None:
        rmtree(sv.temp_working_dir)
    new_sv = new_session_variables()
    # Order of output: prompt_edit, chat_history, history_text, history_text_full, sys_prompt_text, model_cards_checkbox, model_cards_text, modality_prompt_dropdown
    return (
        new_sv,
        "Enter your prompt here",
        ChatHistory(),
        HTML_PLACEHOLDER,
        HTML_PLACEHOLDER,
        new_sv.sys_prompt,
        new_sv.use_model_cards,
        new_sv.sys_msg,
        new_sv.modality_prompt,
    )


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


def update_sys_prompt(sys_prompt, sv):
    """Update the system prompt"""
    logger.debug(f"Updating the system prompt")
    sv.sys_prompt = sys_prompt
    return sv


def update_model_cards_text(model_cards, sv):
    """Update the model cards"""
    logger.debug(f"Updating the model cards contents")
    sv.sys_msg = model_cards
    return sv

def update_model_cards_checkbox(use_model_cards, sv):
    """Update the model cards checkbox"""
    logger.debug(f"Updating the model cards checkbox")
    sv.use_model_cards = use_model_cards
    return sv


def update_modality_prompt(modality_prompt, sv):
    """Update the modality prompt"""
    logger.debug(f"Updating the modality prompt")
    sv.modality_prompt = modality_prompt
    return sv


def download_file():
    """Download the file."""
    return [gr.DownloadButton(visible=False)]


def create_demo(source, model_path, conv_mode, server_port):
    """Main function to create the Gradio interface"""
    generator = M3Generator(source=source, model_path=model_path, conv_mode=conv_mode)

    with gr.Blocks(css=CSS_STYLES) as demo:
        gr.HTML(TITLE, label="Title")
        chat_history = gr.State(value=ChatHistory())  # Prompt history
        sv = gr.State(value=SessionVariables())

        with gr.Row():
            with gr.Column():
                image_dropdown = gr.Dropdown(
                    label="Select an image", choices=["Please select .."] + list(IMG_URLS_OR_PATHS.keys())
                )
                image_input = gr.Image(
                    label="Image", sources=[], placeholder="Please select an image from the dropdown list."
                )
                image_slider = gr.Slider(0, 2, 1, 0, visible=False)

                with gr.Accordion("View Parameters", open=False):
                    temperature_slider = gr.Slider(
                        label="Temperature", minimum=0.0, maximum=1.0, step=0.01, value=0.0, interactive=True
                    )
                    top_p_slider = gr.Slider(
                        label="Top P", minimum=0.0, maximum=1.0, step=0.01, value=0.9, interactive=True
                    )
                    max_tokens_slider = gr.Slider(
                        label="Max Tokens", minimum=1, maximum=1024, step=1, value=1024, interactive=True
                    )

                with gr.Accordion("System Prompt and Message", open=False):
                    modality_prompt_dropdown = gr.Dropdown(
                        label="Select Modality",
                        choices=["Unknown", "Auto", "CT", "MRI", "CXR"],
                    )
                    model_cards_checkbox = gr.Checkbox(
                        label="Use Model Cards",
                        value=sv.value.use_model_cards,
                        info="Check this to include the model cards of the experts."
                    )
                    model_cards_text = gr.Textbox(label="Model Cards", value=sv.value.sys_msg, lines=8)
                    sys_prompt_text = gr.Textbox(
                        label="System Prompt",
                        value=sv.value.sys_prompt,
                        lines=4,
                    )

            with gr.Column():
                with gr.Tab("In front of the scene"):
                    history_text = gr.HTML(HTML_PLACEHOLDER, label="Previous prompts", max_height=600)
                with gr.Tab("Behind the scene"):
                    history_text_full = gr.HTML(HTML_PLACEHOLDER, label="Previous prompts full", max_height=600)
                image_download = gr.DownloadButton("Download the file", visible=False)
                clear_btn = gr.Button("Clear Conversation")
                with gr.Row(variant="compact"):
                    prompt_edit = gr.Textbox(
                        label="TextPrompt",
                        container=False,
                        placeholder="Please ask a question about the current image or 2D slice",
                        scale=2,
                    )
                    submit_btn = gr.Button("Submit", scale=0)
                examples = gr.Examples([[""]], prompt_edit)

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
            inputs=[image_dropdown, sv],
            outputs=[image_input, sv, image_slider, examples.dataset],
        )
        image_slider.release(
            fn=update_image_selection,
            inputs=[image_dropdown, sv, image_slider],
            outputs=[image_input, sv, image_slider, examples.dataset],
        )
        temperature_slider.change(fn=update_temperature, inputs=[temperature_slider, sv], outputs=[sv])
        top_p_slider.change(fn=update_top_p, inputs=[top_p_slider, sv], outputs=[sv])
        max_tokens_slider.change(fn=update_max_tokens, inputs=[max_tokens_slider, sv], outputs=[sv])
        sys_prompt_text.change(fn=update_sys_prompt, inputs=[sys_prompt_text, sv], outputs=[sv])
        model_cards_checkbox.change(fn=update_model_cards_checkbox, inputs=[model_cards_checkbox, sv], outputs=[sv])
        model_cards_text.change(fn=update_model_cards_text, inputs=[model_cards_text, sv], outputs=[sv])
        modality_prompt_dropdown.change(fn=update_modality_prompt, inputs=[modality_prompt_dropdown, sv], outputs=[sv])
        # Reset button
        clear_btn.click(
            fn=clear_all_convs,
            inputs=[sv],
            outputs=[
                sv,
                prompt_edit,
                chat_history,
                history_text,
                history_text_full,
                sys_prompt_text,
                model_cards_checkbox,
                model_cards_text,
                modality_prompt_dropdown,
            ],
        )

        # States
        sv.change(
            fn=clear_one_conv,
            inputs=[sv],
            outputs=[
                sv,
                image_input,
                image_dropdown,
                temperature_slider,
                top_p_slider,
                max_tokens_slider,
                image_download,
                image_slider,
            ],
        )
        demo.queue()
        demo.launch(server_name="0.0.0.0", server_port=server_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Add the argument to load multiple models from a JSON file
    parser.add_argument(
        "--convmode",
        type=str,
        default="vicuna_v1",
        help="The conversation mode to use. Required if source is 'local'.",
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        default="/data/checkpoints/vila-m3-8b",
        help="The path to the model to load. Required if source is 'local'.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="The port to run the Gradio server on.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        help="The source of the model. Option is 'huggingface' or 'local'.",
    )
    args = parser.parse_args()
    cache_images()
    create_demo(args.source, args.modelpath, args.convmode, args.port)
    cache_cleanup()
