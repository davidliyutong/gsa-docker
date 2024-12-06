ARG BASE_IMAGE="gsa-docker:v0"
FROM ${BASE_IMAGE}

# copy models
RUN mkdir -p /root/.cache/huggingface/hub
COPY ./checkpoints/models--bert-base-uncased/ /root/.cache/huggingface/hub/models--bert-base-uncased/ 
COPY ./checkpoints/models--runwayml--stable-diffusion-inpainting/ /root/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/
COPY ./checkpoints/models--Salesforce--blip-image-captioning-large/ /root/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-large/

# modify the app
COPY app/fastapi/app.py /home/appuser/Grounded-Segment-Anything/app.py

ENV TRANSFORMERS_OFFLINE=1

ENTRYPOINT [ "/bin/bash" ]
CMD [ "-c", "python app.py" ]