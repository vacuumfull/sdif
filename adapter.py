import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image


class AdapterClass:
    def __init__(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")
        self.pipeline.set_ip_adapter_scale(0.5)
        self.generator = torch.Generator(device="gpu").manual_seed(26)
        self.image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein_base.png")
        
    
    def prompt(self, text: str):
        image = self.pipeline(
            prompt=text,
            ip_adapter_image=self.image,
            negative_prompt="lowres, bad anatomy, worst quality, low quality",
            num_inference_steps=100,
            generator=self.generator,
        ).images[0]
        return image