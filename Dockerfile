FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=0
ARG TORCH_ARCH=
ARG SAM_CKPT_PATH="./checkpoints/sam_vit_h_4b8939.pth"
ARG GROUNDED_DINO_CKPT_PATH="./checkpoints/groundingdino_swint_ogc.pth"

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-11.6/

RUN mkdir -p /home/appuser/Grounded-Segment-Anything
COPY ./Grounded-Segment-Anything /home/appuser/Grounded-Segment-Anything/

RUN sed -i s@/archive.ubuntu.com/@/mirror.sjtu.edu.cn/@g /etc/apt/sources.list && \
    sed -i s@/security.ubuntu.com/@/mirror.sjtu.edu.cn/@g /etc/apt/sources.list && \
    sed -i s@/ports.ubuntu.com/@/mirror.sjtu.edu.cn/@g /etc/apt/sources.list

RUN apt-get update && apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* nano=2.* \
    vim=2:* -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

WORKDIR /home/appuser/Grounded-Segment-Anything
RUN python -m pip install --no-cache-dir -e segment_anything

# When using build isolation, PyTorch with newer CUDA is installed and can't compile GroundingDINO
RUN python -m pip install --no-cache-dir wheel
RUN python -m pip install --no-cache-dir --no-build-isolation -e GroundingDINO

WORKDIR /home/appuser
RUN pip install --no-cache-dir diffusers[torch]==0.15.1 opencv-python==4.7.0.72 \
    pycocotools==2.0.6 matplotlib==3.5.3 \
    onnxruntime==1.14.1 onnx==1.13.1 ipykernel==6.16.2 scipy gradio==3.50.2 litellm==1.42.5 openai

COPY ${SAM_CKPT_PATH} /home/appuser/Grounded-Segment-Anything/
COPY ${GROUNDED_DINO_CKPT_PATH} /home/appuser/Grounded-Segment-Anything/

# modify the app
COPY app.py /home/appuser/Grounded-Segment-Anything/app.py
COPY bootstrap.py /home/appuser/Grounded-Segment-Anything/
WORKDIR /home/appuser/Grounded-Segment-Anything
# RUN python bootstrap.py

ENTRYPOINT [ "/bin/bash" ]
CMD [ "-c", "python gradio_app.py" ]