FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
COPY . /scan
WORKDIR /scan
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN ["pip", "install", "-r", "./requirements.txt"]
# ENTRYPOINT ["python", "simclr.py", "--config_env", "configs/env.yml", "--config_exp", "configs/pretext/simclr_cifar10.yml"]
