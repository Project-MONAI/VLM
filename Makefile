demo_m3:
	cd thirdparty/VILA; \
	./environment_setup.sh
	pip install -U python-dotenv deepspeed gradio monai[nibabel,pynrrd,skimage,fire,ignite] torchxrayvision huggingface_hub
	mkdir -p $(HOME)/.torchxrayvision/models_data/ \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O $(HOME)/.torchxrayvision/models_data/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O $(HOME)/.torchxrayvision/models_data/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O $(HOME)/.torchxrayvision/models_data/mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/mimic_nb-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O $(HOME)/.torchxrayvision/models_data/mimic_nb-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O $(HOME)/.torchxrayvision/models_data/nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/pc-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O $(HOME)/.torchxrayvision/models_data/pc-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    -O $(HOME)/.torchxrayvision/models_data/kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    && wget https://github.com/mlmed/torchxrayvision/releases/download/v1/pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt \
    -O $(HOME)/.torchxrayvision/models_data/pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt; \
	mkdir -p $(HOME)/.cache/torch/hub/bundle \
	&& python -m monai.bundle download vista3d --version 0.5.4 --bundle_dir $(HOME)/.cache/torch/hub/bundle \
	&& unzip $(HOME)/.cache/torch/hub/bundle/vista3d_v0.5.4.zip -d $(HOME)/.cache/torch/hub/bundle/vista3d_v0.5.4
