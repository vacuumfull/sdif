import torch
import os
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoPipelineForText2Image
from diffusers.utils import load_image
from io import BytesIO
import base64


MODEL_PATH = '../stable-diffusion-webui/models/Stable-diffusion/'
IMAGE_URL ='./images/'
NEGATIVE_PROMPT = '(deformed, distorted, disfigured:1.3), poorly drawn,bad anatomy, wrong anatomy, extralimb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation'
NUM_STEPS = 25

class ImageSd:
    def __init__(self, filename: str = None):
        self.filename = f'{filename}.jpg' if filename else None

    def save(self, image):
        image.save(f'{IMAGE_URL}{self.filename}')
        return self.filename

    def remove(self, filename):
        dist = f'{IMAGE_URL}{filename}'
        is_exists = os.path.exists(dist)
        if is_exists:
            os.remove(dist)
        return is_exists

class StableDif:
    def __init__(self, use_adapter = False, uid = None):
        self.model_name = "Reliberate_v3"
        self.user_id = base64.b64encode(f'{uid}'.encode()).decode() if uid else None
        self.image_sd = ImageSd(self.user_id)
     
        self.model_path = f"{MODEL_PATH}{self.model_name}.safetensors"
        self.pipeline = StableDiffusionPipeline.from_single_file(
            self.model_path,
            generator=torch.Generator(device="cpu").manual_seed(26),
            torch_dtype=torch.float16,
            use_safetensors=True,
            num_inference_steps=NUM_STEPS,
            negative_prompt=NEGATIVE_PROMPT,
            denoising_strength = 0.7
        ).to("cuda")
        self.pipeline_text_to_image = StableDiffusionPipeline.from_single_file(
            self.model_path, 
            torch_dtype=torch.float16,
            generator=torch.Generator(device="cpu").manual_seed(26),
            variant="fp16",
            use_safetensors=True,
            num_inference_steps=NUM_STEPS,
            negative_prompt=NEGATIVE_PROMPT,
            denoising_strength = 0.7
        ).to("cuda")
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        if use_adapter:
            self.pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
            self.pipeline.set_ip_adapter_scale(0.5)


    def prompt(self, text: str):
        image = self.pipeline_text_to_image(prompt=text).images[0]
        return image
    
    def save_image(self, image):
        return self.image_sd.save(image)

    def delete_image(self, filename):
        return self.image_sd.remove(filename)
