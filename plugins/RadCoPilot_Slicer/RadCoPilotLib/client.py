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

import cgi
import http.client
import json
import logging
import re
from pathlib import Path
from urllib.parse import quote_plus, unquote, urlencode, urlparse

import requests

logger = logging.getLogger(__name__)


class RadCoPilotClient:
    """Basic RadCoPilot Client to invoke infer API over http/https."""

    def __init__(self, server_url=None, tmpdir=None, client_id=None):
        """:param server_url: Server URL for RadCoPilot. (e.g. http://127.0.0.1:8000).

        :param tmpdir: Temp directory to save temporary files.  If None then it uses tempfile.tempdir
        :param client_id: Client ID that will be added for all basic requests
        """
        self._server_url = server_url.rstrip("/").strip() if server_url is not None else server_url
        # self._tmpdir = tmpdir if tmpdir else tempfile.tempdir if tempfile.tempdir else "/tmp"
        # self._client_id = client_id
        # self._headers = {}

    def get_server_url(self):
        """Return server url.

        :return: the url for RadCoPilot server
        """
        return self._server_url

    def set_server_url(self, server_url):
        """Set url for RadCoPilot server.

        :param server_url: server url for RadCoPilot
        """
        self._server_url = server_url.rstrip("/").strip()

    def info(self):
        """Invoke /info/ request over RadCoPilot Server.

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

    def getAnswer(self, inputText, volumePath):
        """Invoke request over RadCoPilot Server.

        :return: json response
        """
        selector = "/v1/chat/completions/"
        url = f"{self._server_url}{selector}"

        headers = {
            "Accept": "text/event-stream"
        }

        stream = False  # Set to True if you want a streaming response

        # Prepare query parameters
        params = {
            "Prompt": inputText,
            "stream": stream  # Pass as a boolean
        }

        # Open the file in binary mode
        with open(volumePath, "rb") as file:
            files = {"file": (volumePath, file, "application/octet-stream")}
            response = requests.post(url, headers=headers, params=params, files=files)
            print(response.json())

        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Error: {response.status_code} - {response.text}")
            return None


class RadCoPilotUtils:
    """Utils class for RadCoPilot client."""
    @staticmethod
    def http_method(method, server_url, selector, body=None, headers=None, content_type=None):
        logging.debug(f"{method} {server_url}{selector}")

        parsed = urlparse(server_url)
        path = parsed.path.rstrip("/")
        selector = path + "/" + selector.lstrip("/")
        logging.debug(f"URI Path: {selector}")

        parsed = urlparse(server_url)
        if parsed.scheme == "https":
            logger.debug("Using HTTPS mode")
            # noinspection PyProtectedMember
            conn = http.client.HTTPSConnection(parsed.hostname, parsed.port, context=ssl._create_unverified_context())
        else:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port)

        headers = headers if headers else {}
        if body:
            if not content_type:
                if isinstance(body, dict):
                    body = json.dumps(body)
                    content_type = "application/json"
                else:
                    content_type = "text/plain"
            headers.update({"content-type": content_type, "content-length": str(len(body))})

        conn.request(method, selector, body=body, headers=headers)
        return RadCoPilotUtils.send_response(conn)

    @staticmethod
    def http_upload(method, server_url, selector, fields, files, headers=None):
        logging.debug(f"{method} {server_url}{selector}")

        url = server_url.rstrip("/") + "/" + selector.lstrip("/")
        logging.debug(f"URL: {url}")

        files = [("files", (os.path.basename(f), open(f, "rb"))) for f in files]
        headers = headers if headers else {}
        response = (
            requests.post(url, files=files, headers=headers)
            if method == "POST"
            else requests.put(url, files=files, data=fields, headers=headers)
        )
        return response.status_code, response.text, None

    @staticmethod
    def http_multipart(method, server_url, selector, fields, files, headers={}):
        logging.debug(f"{method} {server_url}{selector}")

        content_type, body = RadCoPilotUtils.encode_multipart_formdata(fields, files)
        headers = headers if headers else {}
        headers.update({"content-type": content_type, "content-length": str(len(body))})

        parsed = urlparse(server_url)
        path = parsed.path.rstrip("/")
        selector = path + "/" + selector.lstrip("/")
        logging.debug(f"URI Path: {selector}")

        if parsed.scheme == "https":
            logger.debug("Using HTTPS mode")
            # noinspection PyProtectedMember
            conn = http.client.HTTPSConnection(parsed.hostname, parsed.port, context=ssl._create_unverified_context())
        else:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port)

        conn.request(method, selector, body, headers)
        return RadCoPilotUtils.send_response(conn, content_type)

    @staticmethod
    def send_response(conn, content_type="application/json"):
        response = conn.getresponse()
        logging.debug(f"HTTP Response Code: {response.status}")
        logging.debug(f"HTTP Response Message: {response.reason}")
        logging.debug(f"HTTP Response Headers: {response.getheaders()}")

        response_content_type = response.getheader("content-type", content_type)
        logging.debug(f"HTTP Response Content-Type: {response_content_type}")

        if "multipart" in response_content_type:
            if response.status == 200:
                form, files = RadCoPilotUtils.parse_multipart(response.fp if response.fp else response, response.msg)
                logging.debug(f"Response FORM: {form}")
                logging.debug(f"Response FILES: {files.keys()}")
                return response.status, form, files, response.headers
            else:
                return response.status, response.read(), None, response.headers

        logging.debug("Reading status/content from simple response!")
        return response.status, response.read(), None, response.headers

    @staticmethod
    def save_result(files, tmpdir):
        for name in files:
            data = files[name]
            result_file = os.path.join(tmpdir, name)

            logging.debug(f"Saving {name} to {result_file}; Size: {len(data)}")
            dir_path = os.path.dirname(os.path.realpath(result_file))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            with open(result_file, "wb") as f:
                if isinstance(data, bytes):
                    f.write(data)
                else:
                    f.write(data.encode("utf-8"))

            # Currently only one file per response supported
            return result_file

    @staticmethod
    def encode_multipart_formdata(fields, files):
        limit = "----------lImIt_of_THE_fIle_eW_$"
        lines = []
        for key, value in fields.items():
            lines.append("--" + limit)
            lines.append('Content-Disposition: form-data; name="%s"' % key)
            lines.append("")
            lines.append(value)
        for key, filename in files.items():
            lines.append("--" + limit)
            lines.append(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"')
            lines.append("Content-Type: %s" % RadCoPilotUtils.get_content_type(filename))
            lines.append("")
            with open(filename, mode="rb") as f:
                data = f.read()
                lines.append(data)
        lines.append("--" + limit + "--")
        lines.append("")

        body = bytearray()
        for line in lines:
            body.extend(line if isinstance(line, bytes) else line.encode("utf-8"))
            body.extend(b"\r\n")

        content_type = "multipart/form-data; boundary=%s" % limit
        return content_type, body

    @staticmethod
    def get_content_type(filename):
        return mimetypes.guess_type(filename)[0] or "application/octet-stream"

    @staticmethod
    def parse_multipart(fp, headers):
        fs = cgi.FieldStorage(
            fp=fp,
            environ={"REQUEST_METHOD": "POST"},
            headers=headers,
            keep_blank_values=True,
        )
        form = {}
        files = {}
        if hasattr(fs, "list") and isinstance(fs.list, list):
            for f in fs.list:
                logger.debug(f"FILE-NAME: {f.filename}; NAME: {f.name}; SIZE: {len(f.value)}")
                if f.filename:
                    files[f.filename] = f.value
                else:
                    form[f.name] = f.value
        return form, files

    @staticmethod
    def urllib_quote_plus(s):
        return quote_plus(s)

    @staticmethod
    def get_filename(content_disposition):
        file_name = re.findall(r"filename\*=([^;]+)", content_disposition, flags=re.IGNORECASE)
        if not file_name:
            file_name = re.findall('filename="(.+)"', content_disposition, flags=re.IGNORECASE)
        if "utf-8''" in file_name[0].lower():
            file_name = re.sub("utf-8''", "", file_name[0], flags=re.IGNORECASE)
            file_name = unquote(file_name)
        else:
            file_name = file_name[0]
        return file_name