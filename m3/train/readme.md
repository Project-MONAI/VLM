# Training Recipe For M3

The M3-VILA is based on [VILA-v1.5](https://github.com/NVlabs/VILA).

There are three model variant types that M3 has been trained on:

1.) VILA1.5-3B

2.) Llama-3-VILA1.5-8B

3.) VILA1.5-13B

For each model variant type, different large language models (LLM's) and vision encoders used, the model training recipe (hyper-parameter configurations) is dependent on the model variant type.

Please address the below in the training bash scripts before trying to execute them:

#### Define paths (replace with actual paths)
1.) STAGE2_PATH="/path/to/your/model", these are VILA pre-trained checkpoints, the link to download them can be found [here](https://github.com/NVlabs/VILA#pre-trained-models).

2.) OUTPUT_DIR="/path/to/output/checkpoints"

3.) CONTAINER_IMAGE="/path/to/your/container/image.sqsh"

4.) CONTAINER_MOUNTS="/path/to/mounts"

5.) WANDB_API_KEY="your_wandb_api_key", [weights and biases](https://wandb.ai/site/) is quite useful for logging metrics and tracking experiments. 

For the environment setup, please refer to the `installation` instructions in [VILA repository](https://github.com/NVlabs/VILA), we recommend to create a container for consistent runs and to avoid running of environmeht_setup.sh again. If the environment is created just the python command can still be executed from the bash scripts

Please make sure that the correct labels of datasets are used in the bash scripts, an example `datasets_mixture.py` can be referred to in the VILA repository. It is located at `/VILA/llava/data/datasets_mixture.py`. 

Below are examples of how datasets are added to the datasets_mixture.py (Append these at the end of the datasets_mixture.py file):
    
    ```
    radvqa = Dataset(
        dataset_name="radvqa",
        dataset_type="torch",
        data_path="/set/path/to/instruction/json/file",
        image_path="/set/path/to/image/folder",
    )
    add_dataset(vn_radvqa)

    slake = Dataset(
        dataset_name="slake",
        dataset_type="torch",
        data_path="/set/path/to/instruction/json/file",
        image_path="/set/path/to/image/folder",
    )
    add_dataset(slake)

    pathvqa = Dataset(
        dataset_name="pathvqa",
        dataset_type="torch",
        data_path="/set/path/to/instruction/json/file",
        image_path="/set/path/to/image/folder",
    )
    add_dataset(pathvqa)
    ```
