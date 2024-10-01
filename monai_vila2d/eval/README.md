## Evaluation of VILA trained checkpoints


Run a VQA evalulation with custom checkpoint and save results under my_test_eval10 directory

```
sbatch eval_radvqa.slurm /path_to_your_checkpoint/checkpoint-1800 my_test_eval10 llama_3
```

The third argument specifies conv_mode/version: llama_3 for 8B, v1 for 13B, hermes-2 for 40B checkpoints
The results folder name (e.g. my_test_eval10) is here to save all the intermediate and final results under

```
${CODE}/eval/my_test_eval10
```

The "results.json" will have the final accuracy numbers:

```
${CODE}/eval/my_test_eval10/radvqa/results.json
```

The output log file is saved in the current directory (from which you started the slurm job), with the corresponding job name, e.g. job_radvqa_eval.log, which can be useful for debugging.

Run all the supported experiments (this will start several slurm jobs)
```
bash ./eval_all.sh /path_to_your_checkpoint/checkpoint-1800 your_name_1800 hermes-2
```

## ENV vars

All evaluations assumes several env vars are set in advance (ofr by editing eval_all.sh)

- CONTAINER - path to VILA container
- DATSETS - path to the main folder with all datasets, mounted as /data/datasets
- CODE - path to the folder with VILA codebase, mounted as /data/code.  The output folder will be saved under /data/code/eval

if no env vars are set, the defaults are used (see eval_all.sh)
