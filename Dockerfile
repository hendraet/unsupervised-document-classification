FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
COPY . /scan
WORKDIR /scan
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "nano", "wget", "-y"]
RUN ["apt-get", "install", "ffmpeg", "libsm6", "libxext6", "-y"]
RUN ["wget", "-L", "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"]
RUN ["pip", "install", "-r", "./requirements.txt"]
# ENTRYPOINT ["python", "simclr.py", "--config_env", "configs/env.yml", "--config_exp", "configs/pretext/simclr_cifar10.yml"]
