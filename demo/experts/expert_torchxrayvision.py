import requests
import os


from experts.base_expert import BaseExpert


class ExpertTXRV(BaseExpert):
    NIM_CXR = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/5ac6a152-6d3f-4691-aef8-9e656557ee45"

    def __init__(self) -> None:
        self.model_name = "CXR"


    def classification_to_string(self, outputs):
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
            ["The resulting predictions are:"] + formatted_items + ["."])


    def is_mentioned(self, input: str):
        """
        Check if the CXR model is mentioned in the input string.

        Args:
            input (str): Text from the LLM, e.g. "Let me trigger <CXR>."

        Returns:
            bool: True if the CXR model is mentioned, False otherwise.
        """

        return self.model_name in str(input)


    def run(self, image_url: str = "", prompt: str = "", **kwargs):
        """
        Run the CXR model to classify the image.

        Args:
            image_url (str): The image URL.
            prompt: the original prompt.
        
        Returns:
            tuple: The classification string, file path, and the next step instruction.
        """

        api_key = os.getenv("api_key", "Invalid")
        if api_key == "Invalid":
            raise ValueError("API key not found. Please set the 'api_key' environment variable.")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
        }
        response = requests.post(self.NIM_CXR, headers=headers, json={"image": image_url})
        response.raise_for_status()
        return self.classification_to_string(response.json()), None, "Use this result to respond to this prompt:\n" + prompt
