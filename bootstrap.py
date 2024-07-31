# This file downloads the required models from hugging face
import torch

## diffusion
try:
    from diffusers import StableDiffusionInpaintPipeline
    inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
                )
    del inpaint_pipeline
    print("Diffusion model downloaded successfully")
except Exception as e:
    print("Diffusion model download failed")
    print(e)



# blip
from transformers import BlipProcessor, BlipForConditionalGeneration
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16)
del blip_processor, blip_model
print("Blip model downloaded successfully")
