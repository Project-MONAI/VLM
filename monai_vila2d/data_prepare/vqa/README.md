Please refer the `instruction_files` folder for the pre-generated instruction data files for each dataset. 

The below commands are for reproducing the instruction data files and are not necessary to run.
### RadVQA
Example command to run generate the instruction training data json file for RadVQA dataset:

```
python radvqa_instruct_data_generate.py \
    --input_json /path/to/VQA_RAD_Dataset_Public.json \
    --output_json /path/to/output/radvqa_instruct.json \
    --data_type train
```

### SLAKE
Example commands to run generate the instruction training data json file for Slake dataset:

Training Instruction Data:
```
python slake_instruct_data_generate.py \
    --input_paths /path/to/slake_dataset/train.json /path/to/slake_dataset/validate.json \
    --output_path /path/to/output/slake_train_val_instruct.json
```
Testing Instruction Data:
```
python slake_instruct_data_generate.py \
    --input_paths /path/to/slake_dataset/test.json \
    --output_path /path/to/output/slake_test_instruct.json
```

### PathVQA
Example command to run generate the instruction training data json file for PathVQA dataset:

```
python pathvqa_instruction_gen_parquet.py --input_path /path/to/input/parquet/files --output_path /path/to/output/processed/dataset
```
Please make sure that the .csv files were succesfully generated from the prior command before running the next command
```
python pathvqa_instruction_generate.py --input_dir /path/to/output/processed/dataset --output_dir /path/to/output_directory
```

### MIMIC-VQA
