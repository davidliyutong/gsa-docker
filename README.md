# Grounded-Segment-Anything as a Service

This repo packs the [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) model as a service using FastAPI and Docker.

## How to build

### Prerequisites

Make sure you have `sam_vit_h_4b8939.pth` and `groundingdino_swint_ogc.pth` in the `checkpoints` directory. They can be downloaded from [sam_vit_h_4b8939.pth](https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth) and [groundingdino_swint_ogc.pth](https://huggingface.co/ShilongLiu/GroundingDINO/blob/main/groundingdino_swint_ogc.pth)

### Build the docker image

Run the `build_image.sh` script to build the docker image.

```bash
bash build_image.sh
```

To disable CUDA support, set the `USE_CUDA` environment variable to `0`:

```bash
USE_CUDA=0; bash build_image.sh
```

## Run the docker container

Run the `run_container.sh` script to start the docker container.

```bash
docker run --ipc=host --gpus all -it --rm -p 7589:7589 gsa-docker:v0
```

To disable gpu support, remove the `--gpus all` flag and add DEVICE=cpu environment variable:

```bash
docker run --ipc=host -it --rm -p 7589:7589 -e DEVICE=cpu gsa-docker:v0
```

## After initialization, export the container

The container will download models during the inference, export the container to a new image when its ready.

```bash
docker commit <container_id> gsa-docker:v0-ready
# docekr export gsa-docker:v0-ready > gsa-docker-v0-ready.tar
```

## Usage

### WebUI

Open the browser and navigate to `http://localhost:7589/` to access the WebUI of Gradio. The Gradio version is currently `3.50.2`.

### API

The API endpoint is `http://localhost:7589/predict`. The API accepts a POST request with the following JSON payload:

```json
{
  "text": "A person is standing on the beach."
}
```