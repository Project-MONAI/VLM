demo_monai_vila2d:
	cd thirdparty/VILA; \
	./environment_setup.sh
	pip install -U python-dotenv deepspeed gradio monai[nibabel,pynrrd,skimage]
