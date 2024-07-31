#!/bin/bash
BASE_IMAGE=gsa-docker:v0-ready

docker build -f manifests/fastapi/Dockerfile --build-arg BASE_IMAGE=$BASE_IMAGE -t gsa-docker-fastapi:v0 .