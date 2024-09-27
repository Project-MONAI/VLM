# Expert data preparation

# 1. Prepare expert training data for VISTA3D 

We can take existing CT datasets, run VISTA3D inference on it and use the results to generate training conversation for M3.

```commandline
export PYTHONPATH=${PWD}/..
ROOT_DIR=../../data/experts/vista3d/inference_results
OUT_FILEPREFIX="../../data/experts/vista3d/llama_gen_expert_data_vista3d_what"
python expert_train_data_cxr.py --in_datapath ${IN_DATAPATH} --root_dir ${ROOT_DIR} --out_fileprefix ${OUT_FILEPREFIX}
```
