import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import os, json
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
prompt = "Cartoon head with a big smile, and sunglasses, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 7.5 # min: 1, max: 14

## IMAGE TO IMAGE
num_images = 3
input_image = None
# Comment to disable image to image
image_path = "../input_images/arms_crossed.jpg"
input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
# Range: 0 to 1
strength = 0.8

## SAMPLER

sampler = "ddpm"
num_inference_steps = 45
seed = 1939591103

output_images = pipeline.generate(
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
    num_of_images=num_images
)

# Combine the input image and the output image into a single image.
import time
timestamp = str(int(time.time()))
for i, output_image in enumerate(output_images):
    os.makedirs(f"../output_images/{timestamp}", exist_ok=True)
    Image.fromarray(output_image).save(f"../output_images/{timestamp}/{i}_final.png")

# Save output image with metadata
metadata = {
    "prompt": prompt,
    "uncond_prompt": uncond_prompt,
    "strength": strength,
    "do_cfg": do_cfg,
    "cfg_scale": cfg_scale,
    "sampler": sampler,
    "num_inference_steps": num_inference_steps,
    "seed": seed,
    "num_of_images": len(output_images)
}
metadata_filename = f"../output_images/{timestamp}/metadata.json"
with open(metadata_filename, 'w') as f:
    json.dump(metadata, f, indent=4)
