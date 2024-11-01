### Integrating expert model data with torchxrayvision

### Prerequisit: Data download & install dependencies
Follow the links for CXR datasets this [README.md](../../README.md#chest-x-ray-classification-datasets-for-model-evaluation). 

For both training and inference, `pip install` packages `torchxrayvision` (for the chest X-ray models), `monai` (for json files writing) and `scikit-image` (for image reading)
are required.  The steps were tested with `torchxrayvision==1.2.4`, `monai==1.3.2` and `scikit-image==0.24.0`.
The corresponding image data are described in the `data_prepare` folder's [readme file](../../README.md).


#### Generate classification scores using all models

```bash
python torchxray_infer.py /data/datasets/mimic-cxr/images -out_dir /data/datasets/mimic-cxr/torchxrayvision/ -cuda -weights densenet121-res224-all
python torchxray_infer.py /data/datasets/mimic-cxr/images -out_dir /data/datasets/mimic-cxr/torchxrayvision/ -cuda -weights densenet121-res224-chex
python torchxray_infer.py /data/datasets/mimic-cxr/images -out_dir /data/datasets/mimic-cxr/torchxrayvision/ -cuda -weights densenet121-res224-mimic_ch
python torchxray_infer.py /data/datasets/mimic-cxr/images -out_dir /data/datasets/mimic-cxr/torchxrayvision/ -cuda -weights densenet121-res224-mimic_nb
python torchxray_infer.py /data/datasets/mimic-cxr/images -out_dir /data/datasets/mimic-cxr/torchxrayvision/ -cuda -weights densenet121-res224-nih
python torchxray_infer.py /data/datasets/mimic-cxr/images -out_dir /data/datasets/mimic-cxr/torchxrayvision/ -cuda -weights densenet121-res224-pc
python torchxray_infer.py /data/datasets/mimic-cxr/images -out_dir /data/datasets/mimic-cxr/torchxrayvision/ -cuda -weights densenet121-res224-rsna
python torchxray_infer.py /data/datasets/mimic-cxr/images -out_dir /data/datasets/mimic-cxr/torchxrayvision/ -cuda -weights resnet50-res512-all
```

The commands will generate `.json` files of classification scores for each image and save them at the `torchxrayvision` subfolder with the following structure:

```
torchxrayvision/
├── densenet121-res224-all
├── densenet121-res224-chex
├── densenet121-res224-mimic_ch
├── densenet121-res224-mimic_nb
├── densenet121-res224-nih
├── densenet121-res224-pc
├── densenet121-res224-rsna
└── resnet50-res512-all
```

The format of the generated `json` file is a dictionary with all the torchxrayvision supported classes.

```json
{
    "Atelectasis": "0.3951",
    "Consolidation": "0.0331",
    "Infiltration": "0.1154",
    "Pneumothorax": "0.0220",
    "Edema": "0.0577",
    "Emphysema": "0.0616",
    "Fibrosis": "0.1451",
    "Effusion": "0.0485",
    "Pneumonia": "0.0961",
    "Pleural_Thickening": "0.5033",
    "Cardiomegaly": "0.5094",
    "Nodule": "0.2884",
    "Mass": "0.0881",
    "Hernia": "0.0082",
    "Lung Lesion": "0.2794",
    "Fracture": "0.4208",
    "Lung Opacity": "0.1399",
    "Enlarged Cardiomediastinum": "0.3182"
}
```


#### Make ensemble classification scores

This step reads all the `.json` files generated in the previous step and compute ensemble probabilities for each image.
The output will be saved at the `torchxrayvision/ensemble` subfolder.

```bash
python make_ensemble_probs.py
```

####  Make VILA format training file

```bash
python make_mimic_expert_json.py
```

For more details about the dataset json file:
please see https://github.com/NVlabs/VILA/tree/main/data_prepare


The data split files and classification labels `mimic-cxr-2.0.0-split.csv` and `mimic-cxr-2.0.0-merged-chexpert.csv`
are from the MIMIC-CXR dataset.
