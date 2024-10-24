# MONAI-VILA: Data Preparation

Preparing the datasets for VILA training and testing requires three steps:
1. Downloading all the datasets (Information to download each dataset is provided in the readme.md for the `vqa`, `report` and `expert` directories)
2. Generating the instruction data for all datasets (Information to generate the instruction data is provided in the readme.md for the `vqa`, `report` and `expert` directory)
3. Adding the prepared datasets to VILA in a data mixture (More information can be found in the [quickstart guide](../train/readme.md))

### VQA Datasets
- PathVQA: Pathology-based VQA dataset with ~4,000 images and ~32,000 QA pairs, focusing on microscopic views of human tissue.
- RadVQA: Radiology VQA dataset containing ~7,000 images and ~25,000 QA pairs, covering various imaging modalities like X-rays and CT scans.
- SLAKE: Specialized medical VQA dataset with ~14,000 images and ~45,000 QA pairs, emphasizing anatomy, modality, and abnormality questions.
- Medical-Diff-VQA: Medical-Diff-VQA dataset, a derivative of the MIMIC-CXR dataset, consists of questions categorized into seven categories: abnormality, location, type, level, view, presence, and difference. We currently exclude the difference category in our training preparation.

### Report Generation Datasets

- MIMIC-CXR-JPG: The MIMIC-CXR-JPG Database v2.0.0 is a publicly available dataset containing 377,110 chest X-ray images in JPG format, along with structured labels derived from 227,827 radiology reports. The dataset is a processed version of MIMIC-CXR, with removed protected health information (PHI) to comply with HIPAA regulations. Its purpose is to support medical research in image understanding, natural language processing, and decision support, providing a standard reference for data splits and image labels.

### Chest X-ray Classification Datasets (for model evaluation)

- ChestXRay14
Diverse and high-quality labeled chest x-ray images with findings positive for pneumothorax, opacity, nodule or mass, or fracture from Majkowska and Mittal et al. Chest Radiograph Interpretation with Deep Learning Models: Assessment with Radiologist-adjudicated Reference Standards and Population-adjusted Evaluation, [Radiology 2020 294:2, 421-431](https://pubs.rsna.org/doi/10.1148/radiol.2019191293).

- CheXpert test set
A test set of the CheXpert dataset consisting of 500 studies from 500 patients randomly sampled from the 1000 studies in the report test set. Eight board-certified radiologists individually annotated each of the studies in the test set. Please see also the [github readme](https://github.com/rajpurkarlab/cheXpert-test-set-labels) and [data page](https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c) for more details.

### Expert Model Datasets
experts
  - CT
  - CXR
    - Medical-Diff-VQA
    - MIMIC-CXR-JPG
  - MRI

| Dataset   | QA/Text Pairs  | Images    | Link |
|-----------|-----------|-----------|------|
| PathVQA   | ~32,000   | ~4,000    | [PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) |
| RadVQA    | ~25,000   | ~7,000    | [RadVQA](https://osf.io/89kps/) |
| SLAKE     | ~45,000   | ~14,000   | [SLAKE](https://www.med-vqa.com/slake/) |
| Medical-Diff-VQA | ~429,000  | 129,232  | [MIMIC-VQA](https://physionet.org/content/medical-diff-vqa/1.0.0) |
| MIMIC-CXR-JPG | 270,784 | 270,784 | [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) |
| ChestXRay14 | 1,962 | 1,962 | [nih-chest-xray](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest#additional_labels) |
| CheXpert | 500 | 500 | [cheXpert-test-set-labels](https://github.com/rajpurkarlab/cheXpert-test-set-labels) |
