demo_monai_vila2d: cxr_download
	cd thirdparty/VILA;
	./environment_setup.sh
	pip install -U python-dotenv deepspeed gradio monai[nibabel,pynrrd,skimage] torchxrayvision

cxr_download:
	mkdir -p /$USER/.torchxrayvision/models_data/ \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O /$USER/.torchxrayvision/models_data/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O /$USER/.torchxrayvision/models_data/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O /$USER/.torchxrayvision/models_data/mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/mimic_nb-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O /$USER/.torchxrayvision/models_data/mimic_nb-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O /$USER/.torchxrayvision/models_data/nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/pc-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O /$USER/.torchxrayvision/models_data/pc-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O /$USER/.torchxrayvision/models_data/kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt \
    -O /$USER/.torchxrayvision/models_data/pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt
