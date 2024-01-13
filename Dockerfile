# CPU
# FROM pytorch/pytorch:latest 
# GPU
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Libraries needed for stone (skin tone classification)
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install facenet-pytorch jupyterlab scipy pandas skin-tone-classifier matplotlib torch-summary scikit-learn

EXPOSE 8888