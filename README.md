# VLM

## MONAI-VILA

monai_vila2d

### Models (placeholder)


### Local Demo

#### Prerequisites

1. **Linux Operating System**

1. **CUDA Toolkit 12.2** (with `nvcc`) for [VILA](https://github.com/NVlabs/VILA).

    To verify CUDA installation, run:
    ```bash
    nvcc --version
    ```
    If CUDA is not installed, use one of the following methods:
    - **Recommended** Use the Docker image: `nvidia/cuda:12.2.2-devel-ubuntu22.04`
        ```bash
        docker run -it --rm --ipc host --gpus all --net host \
            -v <ckpts_dir>:/data/checkpoints \
            nvidia/cuda:12.2.2-devel-ubuntu22.04 bash
        ```
    - **Manual Installation (not recommended)**: Download the appropiate package from [NVIDIA offical page](https://developer.nvidia.com/cuda-12-2-2-download-archive)

1. **Python 3.10** and **Git**
    
    To install these, run
    ```bash
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev git
    ```
    NOTE: The commands are tailored for the Docker image `nvidia/cuda:12.2.2-devel-ubuntu22.04`. If using a different setup, adjust the commands accordingly.


#### Setup Environment

1. Clone the repository and set up the environment:
    ```bash
    git clone https://github.com/Project-MONAI/VLM --recursive
    cd VLM
    python -m venv .venv
    source .venv/bin/activate
    make demo_monai_vila2d
    ```

#### Running the Gradio Demo

1. Navigate to the demo directory:
    ```bash
    cd demo
    ```

1. Set the API keys for calling the expert models:
    ```bash
    export api_key=<your nvcf key>
    export NIM_API_KEY=<your NIM key>
    ```

1. Start the Gradio demo:
    ```bash
    python demo/gradio_monai_vila2d.py  \
        --modelpath /data/checkpoints/<checkpoint-name> \
        --convmode <llama_3 or vicuna_1> \
        --port 7860
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
