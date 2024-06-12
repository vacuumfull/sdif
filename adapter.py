import torch
import os
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoPipelineForText2Image
from diffusers.utils import load_image
from io import BytesIO
import base64

torch.cuda.empty_cache()

IMAGE_URL ='./images/'

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
        
        self.user_id = base64.b64encode(f'{uid}'.encode()).decode() if uid else None
        self.image_sd = ImageSd(self.user_id)
        self.generator = torch.Generator(device="cpu").manual_seed(26)
        self.model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipeline_text_to_image = AutoPipelineForText2Image.from_pretrained(
            self.model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
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
        return self.image_sd.delete(filename)


class AdapterClass:
    def __init__(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")
        self.pipeline.set_ip_adapter_scale(0.5)
        self.generator = torch.Generator(device="cpu").manual_seed(26)
        self.pipline_text_to_image = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
        self.image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein_base.png")
        
    
    def prompt_image(self, text: str):
        image = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
        return image
    
    def prompt_image(self, text: str):
        image = self.pipeline_text_to_image(prompt=text).images[0]
        return image

    @staticmethod
    def save_image(image):
        filename = 'test.jpg'
        image.save(f'./{filename}')
        return filename

    @staticmethod
    def remove_image(filename):
        os.remove(f'./{filename}')
