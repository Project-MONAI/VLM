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
import io
import os
import shutil

import pandas as pd
from PIL import Image
from tqdm import tqdm

# Set up argument parser
parser = argparse.ArgumentParser(description='Process PathVQA parquet files.')
parser.add_argument('--input_path', type=str, required=True, help='Root path where all parquet files are located')
parser.add_argument('--output_path', type=str, required=True, help='Root path where processed dataset will be saved')
args = parser.parse_args()

# Use the parsed arguments
parquet_root_path = args.input_path
output_root_path = args.output_path

# Define the splits and their corresponding parquet files
splits = {
    "train": [
        "train-00000-of-00007-f2d0e9ef9f022d38.parquet",
        "train-00001-of-00007-47d8e0220bf6c933.parquet",
        "train-00002-of-00007-7fb5037c4c5da7be.parquet",
        "train-00003-of-00007-74b9b7b81cc55f90.parquet",
        "train-00004-of-00007-77eea90af4a55dce.parquet",
        "train-00005-of-00007-5332ec423be520bd.parquet",
        "train-00006-of-00007-637a58c700b604af.parquet"
    ],
    "val": [
        "validation-00000-of-00003-90a5518d26493b67.parquet",
        "validation-00001-of-00003-cbfe947a3418595c.parquet",
        "validation-00002-of-00003-9ec816895bd3bc20.parquet"
    ],
    "test": [
        "test-00000-of-00003-e9adadb4799f44d3.parquet",
        "test-00001-of-00003-7ea98873fc919813.parquet",
        "test-00002-of-00003-1628308435019820.parquet"
    ]
}

# Iterate over each split
for split, parquet_files in splits.items():
    # Create output directories
    os.makedirs(f"{output_root_path}/{split}/", exist_ok=True)
    
    # Initialize an empty DataFrame to store all data for the split
    split_data = pd.DataFrame()
    
    # Read each parquet file and append to the split_data DataFrame
    for parquet_file in parquet_files:
        file_path = os.path.join(parquet_root_path, parquet_file)
        df = pd.read_parquet(file_path)
        split_data = pd.concat([split_data, df], ignore_index=True)
    
    # Extract image filenames and metadata
    image_bytes = split_data['image'].apply(lambda x: x['bytes'])  # Extract 'bytes' from the dictionary
    image_paths = split_data['image'].apply(lambda x: x['path'])  # Extract 'path' from the dictionary
    questions = split_data['question']
    answers = split_data['answer']
    
    # Save images to the output directory with a progress bar
    for i, (img_bytes, img_path) in tqdm(enumerate(zip(image_bytes, image_paths)), total=len(image_bytes), desc=f"Processing {split} images"):
        # Convert bytes to an image
        image = Image.open(io.BytesIO(img_bytes))
        
        # Convert image to RGB if it's in CMYK
        if image.mode == 'CMYK':
            image = image.convert('RGB')
        
        # Define the destination path
        dst_image_path = f"{output_root_path}/{split}/{img_path}"
        
        # Save the image as a .jpg file
        image.save(dst_image_path, format='JPEG')
    
    # Save the metadata to a CSV file
    metadata = pd.DataFrame({
        'file_name': image_paths,
        'question': questions,
        'answer': answers
    })
    metadata.to_csv(f"{output_root_path}/{split}_metadata.csv", index=False)
