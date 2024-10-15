demo_monai_vila2d:
	cd thirdparty/VILA; \
	./environment_setup.sh
	pip install -U nibabel python-dotenv deepspeed gradio
