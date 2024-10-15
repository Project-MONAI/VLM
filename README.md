# VLM

## MONAI-VILA

monai_vila2d

### Models (placeholder)


### Local Demo

- Make sure you have CUDA 12.2
    - Manually install it: https://developer.nvidia.com/cuda-12-2-2-download-archive
    - (Recommendded) Use Docker image: `nvidia/cuda:12.2.2-devel-ubuntu22.04`
        NOTE: You may need to install `python3.10`, `git` manually in this image.
        ```bash
        docker run -itd --rm --ipc host --gpus all --net host \
            -v /localhome/local-mingxinz:/workspace \
            -w /workspace/nvidia/VLM \
            nvidia/cuda:12.2.2-devel-ubuntu22.04 bash
        apt-get update && apt-get install -y python3.10 python3.10-venv git
        ```

- Prepare the environment:
    ```bash
    cd $HOME && git clone https://github.com/Project-MONAI/VLM
    cd $HOME/VLM
    python3.10 -m venv .venv
    source .venv/bin/activate
    make demo_monai_vila2d
    ```

```bash
python demo/gradio_monai_vila2d.py
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
