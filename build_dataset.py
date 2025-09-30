import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import os
import glob
from tqdm import tqdm

model_path = "./models/instruct-pix2pix"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True
)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

images = []
names = []
for image_path in tqdm(glob.glob(".\\data\\train\\*.jpg"), desc="Loading images"):
    images.append(Image.open(image_path).convert("RGB"))
    names.append(os.path.basename(image_path))


prompts = []
for text_path in tqdm(glob.glob(".\\data\\train\\*.txt"), desc="Loading prompts"):
    with open(text_path, "r", encoding='utf-8') as f:
        prompts.append(f.read().strip())

batch_size = 4
for i in tqdm(range(0, len(images), batch_size)):
    image_batch = images[i:i + batch_size]
    prompt_batch = prompts[i:i + batch_size]
    images_out = pipe(image=image_batch, prompt=prompt_batch, num_inference_steps=40, guidance_scale=7.5).images
    for j, image in enumerate(images_out):
        dims = images[i + j].size
        stretched_image = image.resize(dims, Image.Resampling.LANCZOS)
        stretched_image.save(os.path.join(".\\results", f"result_{names[i + j]}"))

