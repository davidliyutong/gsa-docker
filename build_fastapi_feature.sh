#!/bin/bash
BASE_IMAGE=gsa-docker:v0

# check if huggingface models exist
if [ ! -d "$HOME/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting" ]; then
	echo "Download the runwayml/stable-diffusion-inpainting model from huggingface"
	exit 1
else
    if [ ! -d "./checkpoints/models--runwayml--stable-diffusion-inpainting" ]; then
        cp -r "$HOME/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/" ./checkpoints/
    fi
fi
if [ ! -d "$HOME/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-large" ]; then
	echo "Download the Salesforce/blip-image-captioning-large model from huggingface"
	exit 1
else
    if [ ! -d "./checkpoints/models--Salesforce--blip-image-captioning-large" ]; then
        cp -r "$HOME/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-large/" ./checkpoints/
    fi
fi
if [ ! -d "$HOME/.cache/huggingface/hub/models--bert-base-uncased" ]; then
	echo "Download the bert-base-uncased model from huggingface"
	exit 1
else
    if [ ! -d "./checkpoints/models--bert-base-uncased" ]; then
        cp -r "$HOME/.cache/huggingface/hub/models--bert-base-uncased/" ./checkpoints/
    fi
fi

docker build -f manifests/fastapi/Dockerfile --build-arg BASE_IMAGE=$BASE_IMAGE -t gsa-docker:v0-fastapi .