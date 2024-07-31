#!/bin/bash
USE_CUDA=${USE_CUDA:-"1"}

# check if  Grounded-Segment-Anything directory exists
if [ ! -d "Grounded-Segment-Anything/.git" ]; then
	echo "Clone the Grounded-Segment-Anything repository first"
	exit 1
fi

# check if checkponts exist
if [ ! -f "checkpoints/sam_vit_h_4b8939.pth" ]; then
	echo "Download the sam_vit_h_4b8939.pth file and place it in the checkpoints directory"
	exit 1
fi

# check if checkponts exist
if [ ! -f "checkpoints/groundingdino_swint_ogc.pth" ]; then
	echo "Download the groundingdino_swint_ogc.pth file and place it in the checkpoints directory"
	exit 1
fi


if [ $USE_CUDA == "1" ]; then
	TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
	BUILD_MESSAGE="I will try to build the image with CUDA support"
else
	TORCH_CUDA_ARCH_LIST=""
	BUILD_MESSAGE="CUDA is not supported"
fi


echo $BUILD_MESSAGE
docker build -f manifests/base/Dockerfile --build-arg USE_CUDA=$USE_CUDA --build-arg TORCH_ARCH=$TORCH_CUDA_ARCH_LIST -t gsa-docker:v0 .