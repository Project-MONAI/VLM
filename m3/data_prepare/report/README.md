# Medical Report Data Preparation for VLM Model Training

This README provides tools and scripts to automate the preparation of medical report data for training the VLM model, which is designed for **medical report generation**. The process includes downloading datasets, processing text using a Large Language Model (LLM), and converting the data into a specific format required for model training, focusing on creating high-quality inputs for generating accurate medical reports.

## Table of Contents
- [Overview](#overview)
- [Steps](#steps)
  - [1. Download Datasets](#1-download-datasets)
  - [2. Convert Text Using LLM](#2-convert-text-using-llm)
    - [2.1 Collect a Sample Sentence Pool](#21-collect-a-sample-sentence-pool)
    - [2.2 Use LLM and Sentence Pool to Convert Text](#22-use-llm-and-sentence-pool-to-convert-text)
  - [3. Convert Data to VLM Model Format](#3-convert-data-to-vlm-model-format)

## Overview

In medical AI applications, particularly for **medical report generation**, preparing data properly is crucial for effective model training. This README automates the preparation of datasets by:
1. Downloading paired images and medical reports.
2. Utilizing a Large Language Model (LLM) to refine the text in the reports.
3. Converting both images and processed text into a format suitable for training the VLM model, which focuses on generating high-quality medical reports.

## Steps

### 1. Download Datasets

In the first step, the script will download the required medical datasets. This includes images (e.g., X-rays, MRIs) and their corresponding textual reports (e.g., diagnosis, findings). The dataset is vital for training the VLM model to accurately generate medical reports from image inputs.

The generation of medical reports is grounded in the MIMIC-CXR-JPG database, specializing in chest X-ray imaging.
To refine the quality of the reports and eliminate noise, we utilize an enhanced text version developed by [DCL](https://github.com/mlii0117/DCL), and subsequently apply additional cleansing procedures to further optimize report accuracy.

### 2. Convert Text Using LLM

This step involves two sub-steps where the LLM helps in processing the textual reports to improve the quality of the data used for **medical report generation**:

#### 2.1 Collect a Sample Sentence Pool

Before performing the text conversion, the LLM will be used to analyze the dataset and collect a sample pool of sentences. This pool will consist of commonly occurring phrases or structures found in the medical reports. These standardized phrases guide the text transformation process to ensure consistent input for report generation.

#### 2.2 Use LLM and Sentence Pool to Convert Text

In this step, the LLM uses the previously collected sample sentence pool to convert the original medical report text into a standardized format. The LLM will replace or reformat certain sentences and medical terminology to ensure the text is uniform and ready for training the VLM model to generate reliable and coherent medical reports.

Here's an example of the text before and after conversion:

- **Before conversion**:
    ```text
    The patient shows evidence of moderate cardiomegaly. There is slight haziness in the lower lung fields, likely indicating early stages of pulmonary edema. Recommend echocardiogram for further evaluation.
    ```

- **After conversion**:
    ```text
    The patient has moderate enlargement of the heart. There is mild cloudiness in the lower lung regions, suggesting early pulmonary edema. An echocardiogram is recommended for further assessment.
    ```

In this example, the LLM standardizes medical terms like “cardiomegaly” to “enlargement of the heart” and simplifies other phrases like “slight haziness” to “mild cloudiness,” ensuring consistency and clarity in the text for better model training.

### 3. Convert Data to VLM Model Format

Finally, the processed text and images are converted into a specific format required for the VLM model. This ensures the data is correctly aligned with the model’s input requirements, allowing the model to generate accurate and high-quality medical reports from the prepared data.
