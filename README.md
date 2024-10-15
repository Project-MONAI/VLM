<p align="center">
  <img src="https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/MONAI-logo-color.png" width="30%"/>
</p>

# MONAI Vision Language Models
The repository provides a collection of vision language models, benchmarks, and related applications, released as part of Project [MONAI](https://monai.io) (Medical Open Network for Artificial Intelligence).

# MONAI-VILA

**VILA-M3** is a *vision language model* designed specifically for medical applications. 
It focuses on addressing the unique challenges faced by general-purpose vision-language models when applied to the medical domain.

For details, see [here](./monai_vila2d/README.md).

### Models (placeholder)


### Local Demo

- Make sure you have CUDA 12.2 and Python 3.10 installed
    - (Recommendded) Use Docker image: `nvidia/cuda:12.2.2-devel-ubuntu22.04`
    ```bash
    docker run -itd --rm --ipc host --gpus all --net host -v <mount paths> \
        nvidia/cuda:12.2.2-devel-ubuntu22.04 bash
    ```
    **IMPORTANT**: Install these packages in container too: `apt-get update && apt-get install -y python3.10 python3.10-venv git`
    - Manually install it: https://developer.nvidia.com/cuda-12-2-2-download-archive
- Set up the dependencies
    ```bash
    git clone https://github.com/Project-MONAI/VLM --recursive
    cd VLM
    python3.10 -m venv .venv
    source .venv/bin/activate
    make demo_monai_vila2d
    ```

- Run the Demo
    ```bash
    cd demo
    # keys to call the expert models
    export api_key=<your nvcf key>
    export NIM_API_KEY=<your NIM key>
    python demo/gradio_monai_vila2d.py  \
        --modelpath <path to the checkpoint> \
        --convmode <llama_3 or vicuna_1>
    ```

## Contributing

To lint the code, please install these packages:

```bash
pip install -r requirements-ci.txt
```

Then run the following command:

```bash
isort --check-only --diff .  # using the configuration in pyproject.toml
black . --check  # using the configuration in pyproject.toml
ruff check .  # using the configuration in ruff.toml
```

To auto-format the code, run the following command:

```bash
isort . && black . && ruff format .
```
