import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import os
import time

# Total memory
total_memory = torch.cuda.get_device_properties(0).total_memory

# Allocated memory
allocated_memory = torch.cuda.memory_allocated(0)

# Cached memory
cached_memory = torch.cuda.memory_reserved(0)

print(f"Total GPU Memory: {total_memory / (1024 ** 3)} GB")
print(f"Allocated Memory: {allocated_memory / (1024 ** 3)} GB")
print(f"Cached Memory: {cached_memory / (1024 ** 3)} GB")

### Setting up the Output Directory
script_dir = os.path.dirname(__file__)

# Build the absolute path for the output directory
output_dir = os.path.join(script_dir, '..', 'output_images')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
### Finished Setting up Output Directory

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

## TEXT TO IMAGE

# prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

input_image = None
# Comment to disable image to image
image_path = "../images/dog.jpg"
# input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9

## SAMPLER

sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    models=models, 
    seed=seed,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# Combine the input image and the output image into a single image.
import time
timestamp = str(int(time.time()))
Image.fromarray(output_image).save(f"../output_images/final_{timestamp}.png")
