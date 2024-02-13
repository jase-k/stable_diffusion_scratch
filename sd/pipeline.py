import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
        prompt: str, 
        uncond_prompt: str, # This is the negative prompt see classifier_free_guidance.png
        input_image=None,
        strength=0.8,  # determines how much noise is added to the image
        # If turned off do_cfg you will not be able to guide how closely the image should resemble the prompt because it won't have any random noise to compare it to
        do_cfg=True, # cfg stands for classifier free guidance. 
        cfg_scale=7.5, # How much the model should pay attention to the prompt  -> This is the W in the classifier_free_guidance.png photo
        sampler_name="ddpm",
        n_inference_steps=50,
        models={},
        seed=None,
        device=None,
        idle_device=None, # This is the device that the model will be moved to when it is not being used (i.e. moved to the CPU when the GPU is not being used)
        tokenizer=None
):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)

        if seed is not None:
            generator.manual_seed(seed)
        else:
            generate.seed()
        
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Conver the prompt into toekns using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids

            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)

            # (2, Seq_Len, Dim) * (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else: 
            # Conver it into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            context = clip(tokens)

        to_idle(clip)

        # Sampler is referred to as the Scheduler in the diffusion.png image in images
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image. resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor) # this creates a (512, 512, 3) tensor 3 for each color channel (RGB)

            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1)) # each pixel was between 0 and 255, now it is between -1 and 1

            # >>> y.shape
            # torch.Size([3, 3, 3])
            # >>> y.unsqueeze(0).shape
            # torch.Size([1, 3, 3, 3])
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel) 
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise) # This gets us to the Z box in diffusion.png in the images folder. 

            to_idle(encoder)
        else:
            # If we are doing text-to-image, start with random noise N(0, I)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)


        # This is the Unet Loop
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            print(f"Step {i} of {n_inference_steps} inference steps.")
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # We are making a copy of the input so we can chunk it later to split into conditional and unconditional noise
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2,1,1,1)

            # model_output is the predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_cond # This is the formula found in classifier_free_guidance.png

            # Now we are going to remove the noise predicted by the UNET 
            latents = sampler.step(timestep, latents, model_output)
        
        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True) # scaling the pixel values back to 0-255 to display image as rgb
        images = images.permute(0, 2, 3, 1) # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]

def rescale(x, old_range, new_range, clamp=False):
    """
    Rescales the input tensor from the old range to the new range
    """
    old_min, old_max = old_range
    new_min, new_max = new_range

    x = x - old_min
    x *=  (new_max - new_min) / (old_max - old_min)
    x = x + new_min

    if clamp:
        # a = torch.randn(4)
        # tensor([-1.7120, 0.1734, -0.0478, -0.0922])
        # torch.clamp(a, min=-0.5, max=0.5)
        # tensor([-0.5000, 0.1734, -0.0478, -0.0922])
        x = x.clamp(new_min, new_max) # Clamp: https://pytorch.org/docs/stable/generated/torch.clamp.html
    return x

def get_time_embedding(timestep: int):
    """
    Returns the time embedding for the given timestep. 
    See positional_encoding.png in the images folder
    """
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1,160)
    # [:,None] adds a new axis to the tensor like unsqueeze
    # >>> y.shape
    # torch.Size([3, 3, 3])
    # >>> y[:,None].shape
    # torch.Size([3, 1, 3, 3])
    x = torch.tensor([timestep], dtype=torch.float32)[:,None] * freqs[None]

    # (1, 320)
    x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    return x



