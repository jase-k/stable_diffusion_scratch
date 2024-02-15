from ddpm import DDPMSampler

from PIL import Image
import torch
import numpy as np
import math

generator = torch.Generator()
generator.manual_seed(0)

ddpm_sampler = DDPMSampler(generator)

# How many noise levels to generate
noise_levels = [0, 10, 50, 75, 100, 250, 500, 750]

img = Image.open("../input_images/arms_crossed.jpg")
img_tensor = torch.tensor(np.array(img))
img_tensor = ((img_tensor / 255.0) * 2.0) - 1.0
# Create a batch by repeating the same image many times
batch = img_tensor.repeat(len(noise_levels), 1, 1, 1)

ts = torch.tensor(noise_levels, dtype=torch.int, device=batch.device)
noise_imgs = []
epsilons = torch.randn(batch.shape, device=batch.device)
# Generate a noisifed version of the image for each noise level
for i in range(len(ts)):
    a_hat = ddpm_sampler.alpha_cumprod[ts[i]]
    noise_imgs.append(
        (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
    )


noise_imgs = torch.stack(noise_imgs, dim=0)
noise_imgs = (noise_imgs.clamp(-1, 1) + 1) / 2
noise_imgs = (noise_imgs * 255).type(torch.uint8)

# Convert back to image and display
display_img = Image.fromarray(noise_imgs[7].squeeze(0).numpy(), 'RGB')

import os
import time

# Build the output directory path
output_dir = os.path.join(os.path.dirname(__file__), '..', 'output_images')
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Generate a timestamp for the output image filename
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_filename = f"noisy_image_{timestamp}.png"

# Save the display image with the generated filename
display_img.save(os.path.join(output_dir, output_filename))
print(f"Image saved to {os.path.join(output_dir, output_filename)}")
