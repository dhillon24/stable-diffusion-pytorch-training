import os
import torch
from tqdm import tqdm
import logging

class Diffusion:
  
  def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, img_channels=3, device="cuda"):
    self.noise_steps = noise_steps
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.img_size = img_size
    self.img_channels = img_channels
    self.device = device
    self.beta = self.prepare_noise_schedule().to(device)
    self.alpha = 1. - self.beta
    self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
  def prepare_noise_schedule(self):
    return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
  
  def noise_images(self, x, t):
    sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
    epsilon = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
  def sample_timestamps(self, n):
    return torch.randint(low=1, high=self.noise_steps, size=(n,))
  
  def sample(self, model, n, context, labels, cfg_scale=3, image_shape=None, normalize_output=True):
    logging.info(f"Sampling {n} new images...")
    model.eval()
    with torch.no_grad():
      if image_shape is None:
        image_shape = (n, self.img_channels, self.img_size, self.img_size)
      else:
        image_shape = (n,) + image_shape
      x = torch.randn(image_shape).to(self.device)
      for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
        t = (torch.ones(n) * i).long().to(self.device)
        if context is not None:
          # cond_context, uncond_context = context.chunk(2)
          # predicted_noise = model(x, t, cond_context, labels)
          # uncond_predicted_noise = model(x, t, uncond_context, None)
          # predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
          x = x.repeat(2, 1, 1, 1) # more VRAM needed but faster
          t = t.repeat(2)
          predicted_noise, uncond_predicted_noise = model(x, t, context, labels).chunk(2)
          predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
          x = x[:n]
          t = t[:n]
        else:
          predicted_noise = model(x, t, context, labels)
        noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
        x = self.denoise_images(x, t, noise, predicted_noise)
    model.train()
    if normalize_output:
      x = (x.clamp(-1,1) + 1) / 2
      x = (x * 255).type(torch.uint8)
    return x 

  def denoise_images(self, x, t, noise, predicted_noise):
      alpha = self.alpha[t][:, None, None, None]
      alpha_hat = self.alpha_hat[t][:, None, None, None]
      beta = self.beta[t][:, None, None, None]
      x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
      return x      
    
    
    
    