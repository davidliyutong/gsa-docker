ARG BASE_IMAGE="gsa-docker:v0-ready"
FROM ${BASE_IMAGE}

# modify the app
COPY app/fastapi/app.py /home/appuser/Grounded-Segment-Anything/app.py

ENV TRANSFORMERS_OFFLINE=1

ENTRYPOINT [ "/bin/bash" ]
CMD [ "-c", "python app.py" ]