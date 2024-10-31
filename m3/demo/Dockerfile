FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y wget python3.10 python3.10-venv python3.10-dev git unzip 
RUN apt-get install -y python3-pip

RUN git clone https://github.com/Project-MONAI/VLM --recursive
WORKDIR /VLM
SHELL ["/bin/bash", "-c"] 
RUN python3.10 -m venv .venv
RUN source .venv/bin/activate
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN make demo_m3
