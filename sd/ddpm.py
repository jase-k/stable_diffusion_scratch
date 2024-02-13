import torch
import numpy as np


# This is the 'Scheduler' from the diffusion.png image in images, but referred to as the 'Sampler' in the code
class DDPMSampler:

    def __init__(
        self, 
        generator: torch.Generator,
        num_training_steps: int = 10000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        # Note: ** 0.5 is the same as taking the square root
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) **2
        self.alphas = 1 - self.betas
        # >>> x = torch.linspace(1,10,10)
        # tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
        # >>> y = torch.cumprod(x, 0)
        # tensor([1.0000e+00, 2.0000e+00, 6.0000e+00, 2.4000e+01, 1.2000e+02, 7.2000e+02,
        #         5.0400e+03, 4.0320e+04, 3.6288e+05, 3.6288e+06])
        # [alpha_0, alpha_0 * alpha_1, alpha_0 * alpha_1 * alpha_2, ...]
        self.alpha_cumprod = torch.cumprod(self.alphas, 0) # cumprod is alpha bar in the stable diffusion paper formula

        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps: int = 50):
        self.num_inference_steps = num_inference_steps

        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)

        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        step_ratio = self.num_training_steps // self.num_inference_steps
        prev_t = (timestep - step_ratio)
        return prev_t

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor): 
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_prev_t = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_prev_t = 1 - alpha_prod_prev_t
        current_alpha_t = alpha_prod_t / alpha_prod_prev_t
        current_beta_t = 1 -current_alpha_t

        # Based on forumla 15 in the ddpm.pdf paper -> finding x0
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # Comput the coefficients for pred_original_sample and current sample x_-t (formula 7 from ddpm.pdf)
        pred_original_sample_coeff = (alpha_prod_prev_t ** 0.5 * current_beta_t )/ beta_prod_t
        # order of operations: ((a ** 0.5) * b) / c
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_prev_t / beta_prod_t

        # Compute the predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        variance = 0 
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise

        # N(0, 1) --> N(mu, sigma^2)
        # X = mu + sigma * Z where Z ~ N(0, 1)
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    # According to the formula 6 and 7 in the ddpm.pdf paper
    def _get_variance(self, timestep: int) -> int:
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t ) * current_beta_t

        variance = torch.clamp(variance, min=1e-20)
        return variance


    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)

        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5  # standard deviation
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # According to the ddpm.pdf paper section 2 formula 4
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod) * noise

        return noisy_samples
    
    # This sets the step the scheduler / sampler starts on. We can add an image and only noise it a little bit 
    # and 'trick' the model to think it came up with that image.
    def set_strength(self, strength: float =1.0):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
