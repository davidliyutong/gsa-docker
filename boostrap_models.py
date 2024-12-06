import huggingface_hub
assert huggingface_hub.__version__ == '0.25.2'

from huggingface_hub import snapshot_download

snapshot_download(repo_id="Salesforce/blip-image-captioning-large")
snapshot_download(repo_id="runwayml/stable-diffusion-inpainting")
snapshot_download(repo_id="bert-base-uncased")