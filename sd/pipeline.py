import torch
import os, time
from PIL import Image
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
        tokenizer=None,
        num_of_images=1
):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip.to(device)

        # cfg stands for Classifier Free Guidance
        if do_cfg:
            # Conver the prompt into toekns using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids

            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)

            # (2, Seq_Len, Dim) * (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else: 
            # Conver it into a list of tokens
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
            print(f"Context shape: {context.shape}")
        
        # interleave is needed here instead of repeat because we want the values stacked: 
        # >>> c = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]],[[2,4,6],[8,10,12],[14,16,18]]])
        # >>> c.shape
        # torch.Size([2, 3, 3])
        # >>> c.repeat(2,1,1)
        # tensor([[[ 1,  2,  3],
        #          [ 4,  5,  6],
        #          [ 7,  8,  9]],

        #         [[ 2,  4,  6],
        #          [ 8, 10, 12],
        #          [14, 16, 18]],

        #         [[ 1,  2,  3],
        #          [ 4,  5,  6],
        #          [ 7,  8,  9]],

        #         [[ 2,  4,  6],
        #          [ 8, 10, 12],
        #          [14, 16, 18]]])
        # >>> c.repeat_interleave(2, dim=0)
        # tensor([[[ 1,  2,  3],
        #          [ 4,  5,  6],
        #          [ 7,  8,  9]],

        #         [[ 1,  2,  3],
        #          [ 4,  5,  6],
        #          [ 7,  8,  9]],

        #         [[ 2,  4,  6],
        #          [ 8, 10, 12],
        #          [14, 16, 18]],

        #         [[ 2,  4,  6],
        #          [ 8, 10, 12],
        #          [14, 16, 18]]])
        # torch.Size([4, 3, 3])

        context = context.repeat_interleave(num_of_images, dim=0)

        ###### Checkpoint 1: Save the context tensor to a text file for analysis ######
        context_filename = "../checkpoints/post_clip_output.txt"

        with open(context_filename, "w") as file:
            file.write("# This is the output of the text embedding layer of the CLIP model # \n\n")
            file.write("Your Prompt (below) has been encoded into the following embedding: \n\n" + prompt + "\n\n")
            file.write(str(context.shape) + "\n")
            file.write(str(context.cpu().numpy()))
        print(f"Context tensor saved to {context_filename}")
        ###### Checkpoint END: 1 ######

        to_idle(clip)

        # Sampler is referred to as the Scheduler in the diffusion.png image in images
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        
        latents_shape = (num_of_images, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            # latents_shape = (1, 3, LATENTS_HEIGHT, LATENTS_WIDTH)
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor.convert("RGB")) # this creates a (512, 512, 3) tensor 3 for each color channel (RGB)


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
            
            # Before passing the input_image_tensor to the encoder, ensure it's on the same device as the encoder.
            input_image_tensor = input_image_tensor.to(device)

            # run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise) # This gets us to the Z box in diffusion.png in the images folder. 

            ######### Checkpoint: 2.a Image Encoded #########
            # Save the latents tensor to a text file for analysis
            context_filename = "../checkpoints/post_encoder_output.txt"

            with open(context_filename, "w") as file:
                file.write("# This is the output of the Encoder -> It will be random noise if no image is provided # \n\n")
                file.write(str(latents.shape) + "\n")
                file.write(str(latents.cpu().numpy()))
            print(f"Context tensor saved to {context_filename}")

            # Directly rescale the latents tensor from (-1, 1) to (0, 255) and convert to numpy array for visualization
            visual_latents = rescale(latents, (-1, 1), (0, 255), clamp=True)
            visual_latents = visual_latents.permute(0, 2, 3, 1)  # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
            visual_latents = visual_latents.to("cpu", torch.uint8).numpy()

            # Save the visual representation of latents to a file
            output_image_path = os.path.join("../checkpoints/", f"post_encoder_output.png")
            
            Image.fromarray(visual_latents[0]).save(output_image_path)
            print(f"Latent visualization saved to {output_image_path}")

            ######### Checkpoint END: 2.a Image Encoded #########

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])


            to_idle(encoder)
        else:
            # If we are doing text-to-image, start with random noise N(0, I)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        ######### Checkpoint: 2.b Image Encoded  + Noised #########
        # Save the latents tensor to a text file for analysis
        context_filename = "../checkpoints/post_encoder_noise_output.txt"

        with open(context_filename, "w") as file:
            file.write("# This is the output of the Encoder -> It will be random noise if no image is provided # \n\n")
            file.write(str(latents.shape) + "\n")
            file.write(str(latents.cpu().numpy()))
        print(f"Context tensor saved to {context_filename}")

        # Directly rescale the latents tensor from (-1, 1) to (0, 255) and convert to numpy array for visualization
        visual_latents = rescale(latents, (-1, 1), (0, 255), clamp=True)
        visual_latents = visual_latents.permute(0, 2, 3, 1)  # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        visual_latents = visual_latents.to("cpu", torch.uint8).numpy()

        # Save the visual representation of latents to a file
        
        for i, vl in enumerate(visual_latents):
            output_image_path = os.path.join("../checkpoints/", f"post_encoder_noise_output_{i}.png")
            Image.fromarray(vl).save(output_image_path)
        print(f"Latent visualization saved to {output_image_path}")

        ######### Checkpoint END: 2.b Image Encoded  + Noised #########
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
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond # This is the formula found in classifier_free_guidance.png
            
            ######### Checkpoint: 3.n Unet Step #########
            # Save the latents tensor to a text file for analysis
            context_filename = f"../checkpoints/unets/{i}.txt"

            with open(context_filename, "w") as file:
                file.write("# This is the output of the Encoder -> It will be random noise if no image is provided # \n\n")
                file.write(str(model_output.shape) + "\n")
                file.write(str(model_output.cpu().numpy()))
            print(f"Context tensor saved to {context_filename}")

            # Directly rescale the latents tensor from (-1, 1) to (0, 255) and convert to numpy array for visualization
            visual_latents = rescale(model_output, (-1, 1), (0, 255), clamp=True)
            visual_latents = visual_latents.permute(0, 2, 3, 1)  # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
            visual_latents = visual_latents.to("cpu", torch.uint8).numpy()

            # Save the visual representation of latents to a file
            output_image_path = os.path.join("../checkpoints/", f"../checkpoints/unets/{i}.png")
            
            Image.fromarray(visual_latents[0]).save(output_image_path)
            print(f"Latent visualization saved to {output_image_path}")

            # Now we are going to remove the noise predicted by the UNET 
            latents = sampler.step(timestep, latents, model_output)

            ######### Checkpoint: 4.n Sampler Step #########
            # Save the latents tensor to a text file for analysis
            context_filename = f"../checkpoints/sampler/{i}.txt"

            with open(context_filename, "w") as file:
                file.write("# This is the output of the Encoder -> It will be random noise if no image is provided # \n\n")
                file.write(str(latents.shape) + "\n")
                file.write(str(latents.cpu().numpy()))
            print(f"Context tensor saved to {context_filename}")

            # Directly rescale the latents tensor from (-1, 1) to (0, 255) and convert to numpy array for visualization
            visual_latents = rescale(latents, (-1, 1), (0, 255), clamp=True)
            visual_latents = visual_latents.permute(0, 2, 3, 1)  # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
            visual_latents = visual_latents.to("cpu", torch.uint8).numpy()

            # Save the visual representation of latents to a file
            output_image_path = os.path.join("../checkpoints/", f"../checkpoints/sampler/{i}.png")
            
            Image.fromarray(visual_latents[0]).save(output_image_path)
            print(f"Latent visualization saved to {output_image_path}")

            ######### Checkpoint END: 4.n Sampler Step #########
        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True) # scaling the pixel values back to 0-255 to display image as rgb
        images = images.permute(0, 2, 3, 1) # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.to("cpu", torch.uint8).numpy()
        return images

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
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)



