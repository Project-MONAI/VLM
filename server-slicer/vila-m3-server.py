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
import os
import shutil
import tempfile
import time
from typing import Dict, Union

import requests
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import HttpUrl

# Import VILA M3 utilities
from utils import ChatHistory, ImageCache, M3Generator, SessionVariables

# API Key and Cloud Volume Link
os.environ["api_key"] = "nvapi"

# FastAPI app setup
app = FastAPI(
    title="VILA-M3",
    openapi_url="/openapi.json",
    docs_url=None,
    redoc_url="/docs",
)

infoRouter = APIRouter(
    prefix="/info",
    tags=["App Info"],
    responses={404: {"description": "Not found"}},
)

@infoRouter.get("/", summary="Get App Info")
async def api_app_info():
    '''Get model info.'''
    return "VILA-M3 model"

app.include_router(infoRouter)

fileUploadRouter = APIRouter(
    prefix="/upload",
    tags=["File Upload"],
    responses={404: {"description": "Not found"}}
)

last_received_file = None

@fileUploadRouter.post("/")
async def upload_file(file: Union[UploadFile, str] = File(...)):
    '''Call for uploading file.'''
    global last_received_file
    
    try:
        if file.startswith('http://') or file.startswith('https://'):
            # If file is a URL string, download it

            last_received_file = file

            return {"filename": {file}, "status": "File path received successfully"}

        else:
            # If file is an UploadFile, read its content
            file_content = await file.read()
            file_name = file.filename

        file_ext = "".join(pathlib.Path(file_name).suffixes)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        
        with temp_file as buffer:
            buffer.write(file_content)
        
        last_received_file = temp_file.name

        print(f'filename: {last_received_file}, status: File uploaded successfully')
        
        return {"filename": file_name, "status": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(fileUploadRouter)

chatCompletionsRouter = APIRouter(
    prefix="/v1/chat/completions",
    tags=["Prompt requests"],
    responses={
        404: {"description": "Not found"},
        200: {"description": "OK"}
    }
)

@chatCompletionsRouter.post('/')
async def chat_completions(
    Prompt: str,
    stream: bool,
    file: UploadFile = File(None),
):
    '''Call for chatting with the model.'''
    global last_received_file


    # Initialize VILA M3 components
    cache_dir = "./data"
    os.makedirs(cache_dir, exist_ok=True)
    cache_images = ImageCache(cache_dir)
    cache_images.cache({"Sample 1": last_received_file})

    sv = SessionVariables()
    chat_history = ChatHistory()
    m3 = M3Generator(cache_images)
    
    try:
        print(f"Prompt received: {Prompt}")

        if file:
            file_ext = "".join(pathlib.Path(file.filename).suffixes)
            image_file = tempfile.NamedTemporaryFile(suffix=file_ext).name

            with open(image_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        elif last_received_file:
            image_file = last_received_file
        else:
            raise HTTPException(status_code=400, detail="No file provided and no previous file available")

        # Process prompt with VILA M3
        sv.image_url = last_received_file
        sv.slice_index = 57  # Example slice index (adjust as needed) - This should come from Slicer!!!
        
        sv, chat_history = m3.process_prompt(Prompt, sv, chat_history)
        
        response_message = chat_history.messages[-1] if chat_history.messages else "No response generated"

        if not stream:
            response = {
                'id': 'chatcmpl-' + os.urandom(12).hex(),
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': 'vila-m3',
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': response_message
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            }
            return JSONResponse(content=response)
        else:
            async def generate():
                response_id = 'chatcmpl-' + os.urandom(12).hex()

                yield f'data: {json.dumps({"id": response_id, "content": response_message})}\n\n'
                yield 'data: [DONE]\n\n'

            return StreamingResponse(generate(), media_type='text/event-stream')

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(chatCompletionsRouter)


@app.get("/", include_in_schema=False)
async def custom_swagger_ui_html():
    '''Custom swagger UI.'''
    html = get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title + " - APIs")

    body = html.body.decode("utf-8")
    body = body.replace("showExtensions: true,", "showExtensions: true, defaultModelsExpandDepth: -1,")
    return HTMLResponse(body)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8010)
