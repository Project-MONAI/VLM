<p align="center">
  <img src="https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/MONAI-logo-color.png" width="30%"/>
</p>

# MONAI Vision Language Models
The repository provides a collection of vision language models, benchmarks, and related applications, released as part of Project [MONAI](https://monai.io) (Medical Open Network for Artificial Intelligence).

## VILA-M3

**VILA-M3** is a *vision language model* designed specifically for medical applications. 
It focuses on addressing the unique challenges faced by general-purpose vision-language models when applied to the medical domain.

For details, see [here](./monai_vila2d/README.md).

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
