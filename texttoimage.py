import torch
from diffusers import DiffusionPipeline
from PIL import Image
from IPython.display import display

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16",
    use_safetensors=True
)
pipe = pipe.to(device)

prompt = "A futuristic warrior standing on a cliff, cinematic, ultra detailed, masterpiece"
negative_prompt = "blurry, deformed, low quality"

image = pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
display(image)
image.save("sdxl_image.png")
