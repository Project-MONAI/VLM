<p align="center">
  <img src="https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/MONAI-logo-color.png" width="30%"/>
</p>

# VILA-M3

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](../LICENSE)
[![Model License](https://img.shields.io/badge/MODEL--License-CC_BY--NC--SA--4.0-red.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

[MONAI Huggingface](https://huggingface.co/monai)

## Introduction 

**VILA-M3** is a *vision language model* designed specifically for medical applications. 
It focuses on addressing the unique challenges faced by general-purpose vision-language models when applied to the medical domain. 
The key characteristics of the model include:

1. **Expert Input in Medical Models**: VILA-M3 integrates expert knowledge into the model, 
acknowledging the demands of precision and domain knowledge of the medical field where general-purpose models may fall short.
  
2. **VILA Base Model**: VILA-M3 leverages the strong capabilities of the VILA vision-language model and fine-tunes it on healthcare-specific datasets.

3. **Hybrid Information Fusion**: VILA-M3 can incorporate 2D, 3D, and even 4D information by fusion of expert model results and VLM predictions.

4. **Open-Source MONAI Module**: The model and several fine-tuned checkpoints are released as part of project [MONAI](https://monai.io). 
We provide scripts for data preparation and a standardized module for benchmarking to evaluate the models in various medical imaging tasks.

Below is an overview of the VILA-M3 with expert model integration and feedback. 
The VLM (based on [VILA](https://github.com/NVlabs/VILA)) can select the most appropriate expert model to run given an image and user prompt. 
The resulting expert model output will be fed back to the VLM for generating the final prediction using a back-and-forth conversation.

<p align="center">
  <img src="docs/images/MONAI-VLM_Overview.svg" width="95%"/>
</p>


## 💡 News

- [2024/10/24] We presented VILA-M3 and the VLM module in MONAI at MONAI Day (slides, recording)
- [2024/10/24] Several fine-tuned healthcare checkpoints are released.

## Performance

### VQA Benchmarks
|                   | Average |
|-------------------|---------|
| VILA-M3-3B        |         |
| Llama3-VILA-M3-8B |         |
| VILA-M3-13B       |         |

### Report Generation Benchmarks
|                   | Average |
|-------------------|---------|
| VILA-M3-3B        |         |
| Llama3-VILA-M3-8B |         |
| VILA-M3-13B       |         |

### Classification Benchmarks
|                   | Average |
|-------------------|---------|
| VILA-M3-3B        |         |
| Llama3-VILA-M3-8B |         |
| VILA-M3-13B       |         |


## Demo
An interactive demo is provided in ...

## Data preparation
To prepare the datasets for training and evaluation, follow the instructions in [data_prepare](./data_prepare).

## Training
To replicate our fine-tuning procedure, utilize the provided scripts.

## Evaluation
To evaluate a model on the above benchmarks, follow the instructions in [eval](./eval/README.md)

## 🔒 License

- The code in this repository is released under [Apache 2.0 license](../LICENSE).
- The fine-tuned weights are released under the [CC-BY-NC-SA-4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).

## Citations

```
TBD
```

# Acknowledgement

- Our models are fine-tuned using [VILA code and base models](https://github.com/NVlabs/VILA).
- We thank the data providers of all the healthcare datasets detailed in [data_prepare](./data_prepare).
- The `Medical-Diff-VQA` data preparation and evaluation scripts were contributed by the authors of the [D-RAX paper](https://arxiv.org/abs/2407.02604).
