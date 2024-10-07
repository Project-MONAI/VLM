# MONAI-VILA: Data Preparation

### VQA Datasets
- PathVQA: Pathology-based VQA dataset with ~4,000 images and ~32,000 QA pairs, focusing on microscopic views of human tissue.
- RadVQA: Radiology VQA dataset containing ~7,000 images and ~25,000 QA pairs, covering various imaging modalities like X-rays and CT scans.
- SLAKE: Specialized medical VQA dataset with ~14,000 images and ~45,000 QA pairs, emphasizing anatomy, modality, and abnormality questions.
- Medical-Diff-VQA: Medical-Diff-VQA dataset, a derivative of the MIMIC-CXR dataset, consists of questions categorized into seven categories: abnormality, location, type, level, view, presence, and difference. We currently exclude the difference category in our training preparation.

### Report Generation Datasets

- MIMIC-CXR-JPG: The MIMIC-CXR-JPG Database v2.0.0 is a publicly available dataset containing 377,110 chest X-ray images in JPG format, along with structured labels derived from 227,827 radiology reports. The dataset is a processed version of MIMIC-CXR, with removed protected health information (PHI) to comply with HIPAA regulations. Its purpose is to support medical research in image understanding, natural language processing, and decision support, providing a standard reference for data splits and image labels.

### Chest X-ray Classification Datasets

### Expert Model Datasets
experts
  - CT
  - CXR
  -- asdfas
  - MRI

| Dataset   | QA/Text Pairs  | Images    | Link |
|-----------|-----------|-----------|------|
| PathVQA   | ~32,000   | ~4,000    | [PathVQA](https://github.com/UCSD-AI4H/PathVQA) |
| RadVQA    | ~25,000   | ~7,000    | [RadVQA](https://github.com/abachaa/VQA-Med-2019) |
| SLAKE     | ~45,000   | ~14,000   | [SLAKE](https://github.com/SLAKE-SLAKE/SLAKE) |
| Medical-Diff-VQA | ~429,000  | 129,232  | [MIMIC-VQA](https://physionet.org/content/medical-diff-vqa/1.0.0) |
| MIMIC-CXR-JPG | 270,784 | 270,784 | [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) |
