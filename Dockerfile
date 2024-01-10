FROM pytorch/pytorch:latest

# Libraries needed for stone (skin tone classification)
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install facenet-pytorch jupyterlab scipy pandas skin-tone-classifier matplotlib

EXPOSE 8888