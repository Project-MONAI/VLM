# Expert data preparation

### 1. Prepare expert training data for VISTA3D 
We can take existing CT datasets, run [VISTA3D](https://github.com/Project-MONAI/model-zoo/tree/dev/models/vista3d) inference on it and use the results to generate training conversation for M3.

```commandline
export PYTHONPATH=${PWD}/..
ROOT_DIR=../../data/experts/vista3d/inference_results
OUT_FILEPREFIX="../../data/experts/vista3d/llama_gen_expert_data_vista3d_what"
python expert_train_data_vista3d.py --in_datapath ${IN_DATAPATH} --root_dir ${ROOT_DIR} --out_fileprefix ${OUT_FILEPREFIX}
```

### 2. Prepare expert training data for BRATS 
Here, we take a brain MRI dataset, run the [BRATS segmentation model](https://github.com/Project-MONAI/model-zoo/tree/dev/models/brats_mri_segmentation) inference on it.
The results can be used to generate training conversation for M3.

In our work, we used the MRI data from the [BRATS 2018 challenge](https://www.med.upenn.edu/sbia/brats2018.html).
- Target: 3 tumor subregions
- Task: Segmentation
- Modality: MRI
- Size: 285 3D volumes (4 channels each)

First, you need to run model inference on your dataset. Please follow the instructions from the [MONAI bundle](https://github.com/Project-MONAI/model-zoo/tree/dev/models/brats_mri_segmentation#execute-inference).

Next, extract 2D training slices from the MRI images and inference results
```commandline
export PYTHONPATH=${PWD}/..
DATALIST=./brats_mri_segmentation/datalist.json
INPUT_IMAGE_DIR=/path/to/BRATS2018
INPUT_LABEL_DIR=/path/to/brats_mri_segmentation_eval
OUTPUT_DIR=../../data/experts/brats/slices
python expert_extract_slices_brats.py --datalist_filename ${DATALIST} --input_image_dir ${INPUT_IMAGE_DIR} --input_label_dir ${INPUT_LABEL_DIR} --output_dir ${OUTPUT_DIR}
```

Finally, we combine the example images to generate a training conversation for M3.
```commandline
export PYTHONPATH=${PWD}/..
META_DATA=../../data/experts/brats/slices/extracted_slices_meta.json
ROOT_DIR=../../data/experts/brats/slices
OUT_FILEPREFIX=../../data/experts/brats/llama_gen_expert_data_brats
python expert_train_data_brats.py --in_meta_data ${META_DATA} --images_root ${ROOT_DIR} --out_fileprefix ${OUT_FILEPREFIX}
```

### 2. Prepare expert training data for TorchXRayVision
For details on how to prepare training & evaluation data with an TorchXRayVision expert model ensemble, see [here](./torchxrayvision/README.md).

And run a command similar to this
```commandline
export PYTHONPATH=${PWD}/..
ROOT_DIR=../../data/experts/cxr/inference_results
OUT_FILEPREFIX="../../data/experts/cxr/cxr_expert"
python expert_train_data_cxr.py --in_datapath ${IN_DATAPATH} --root_dir ${ROOT_DIR} --out_fileprefix ${OUT_FILEPREFIX}
```