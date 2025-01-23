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

import json
import logging
import requests

logger = logging.getLogger(__name__)


class RadCoPilotClient:
    """
    Basic RadCoPilot Client to invoke infer API over http/https
    """

    def __init__(self, server_url=None, tmpdir=None, client_id=None):
        """
        :param server_url: Server URL for RadCoPilot (e.g. http://127.0.0.1:8000)
        :param tmpdir: Temp directory to save temporary files.  If None then it uses tempfile.tempdir
        :param client_id: Client ID that will be added for all basic requests
        """

        self._server_url = server_url.rstrip("/").strip() if server_url is not None else server_url
        # self._tmpdir = tmpdir if tmpdir else tempfile.tempdir if tempfile.tempdir else "/tmp"
        # self._client_id = client_id
        # self._headers = {}

    def get_server_url(self):
        """
        Return server url

        :return: the url for monailabel server
        """
        return self._server_url

    def set_server_url(self, server_url):
        """
        Set url for monailabel server

        :param server_url: server url for monailabel
        """
        self._server_url = server_url.rstrip("/").strip()

    def info(self):
        """
        Invoke /info/ request over RadCoPilot Server

        :return: string response
        """
        selector = "/info/"
        url = f"{self._server_url}{selector}"

        headers = {
            "Accept": "text/event-stream"
        }

        response = requests.get(url, headers=headers)
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(f"HTTP Error {response.status_code}: {response.reason}")

        response_text = response.text
        logging.debug(f"Response: {response_text}")
        return response_text  # The API returns a string, so we don't need to parse it as JSON

    def getAnswer(self, inputText):
        """
        Invoke request over RadCoPilot Server

        :return: json response
        """

        selector = "/v1/chat/completions/"
        url = f"{self._server_url}{selector}"

        headers = {
            "Accept": "text/event-stream"
        }

        stream = False  # Set to True if you want a streaming response
        file_path = "/media/andres/disk-workspace/RadCoPilot/nn-tensorrt-llm/examples/multimodal/CTChest.nii.gz"

        # Prepare query parameters
        params = {
            "Prompt": inputText,
            "stream": stream  # Pass as a boolean
        }

        # Open the file in binary mode
        with open(file_path, "rb") as file:
            files = {"file": (file_path, file, "application/octet-stream")}
            response = requests.post(url, headers=headers, params=params, files=files)
            print(response.json())

        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Error: {response.status_code} - {response.text}")
            return None
