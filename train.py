import os
import torch
import yaml
import copy
import logging
import functools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim
from torchvision import utils as vutils
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter      ## TODO: Consider using WandDB
from lightning.fabric import Fabric

from utils import *
from unet import UNETConditionalScalable as UNet
from unet import EMA
from metrics import LPIPS, FID
from vqgan import VQGAN, Discriminator
from ddpm import Diffusion
from transformer import VQGANTransformer
from clip import CLIPModel, clip_loss, clip_metrics

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

torch.set_float32_matmul_precision('medium')
torch.autograd.set_detect_anomaly(True)

class VGGANTransformerTrainer():
  def __init__(self, args):
    self.model = VQGANTransformer(args).to(device=args.device)
    self.optim = self.configure_optimizers()
  
  def configure_optimizers(self):
    decay, no_decay = set(), set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    for mn, m in self.model.transfomer.named_modules():
      for pn, p in m.named_parameters():
        fpn = f"{mn}.{pn}" if mn else pn  # full param name

        if pn.endswith("bias"):
          no_decay.add(fpn)
        elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
          decay.add(fpn)
        elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
          no_decay.add(fpn)

    no_decay.add("pos_emb")
    param_dict = {pn: p for pn, p in self.named_parameters()}

    optim_groups = [
      {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.args.weight_decay},
      {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=self.args.learning_rate, betas=(self.args.beta1, self.args.beta2))

    return optimizer

  def train(self, args):
    train_dataset = load_dataset(args)
    for epoch in range(args.epochs):
      with tqdm(range(len(train_dataset))) as pbar:
        for i, imgs in zip(pbar, train_dataset):
          self.optim.zero_grad()
          imgs = imgs.to(args.device)
          logits, targets = self.model(imgs)
          loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
          loss.backward()
          self.optim.step()
          pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
          pbar.update(0)
        log, sampled_imgs = self.model.log_images(imgs[0][None])
        vutils.save_image(sampled_imgs, os.path.join("results", f"transformer_{epoch}"), nrow=4)
        autoencoder_plot_images(log)
        torch.save(self.model.state_dict(), os.path.join("models", f"transformer_{epoch}.pt"))
        
class VQGANTrainer:
  def __init__(self, args):
    self.args = args
    self.model = None
    self.discriminator = None
    self.perceptual_loss = None
    # self.fid = None
    self.opt_vq = None
    self.opt_disc = None
    self.DeviceManager = functools.partial(DeviceContextManager, 
                                           active_device=args.device, 
                                           idle_device=args.idle_device)
    
  def configure_optimizers(self, args):
    lr = args.learning_rate
    opt_vq = torch.optim.Adam(
      list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) +\
      list(self.model.codebook.parameters()) + list(self.model.quant_conv.parameters()) +\
      list(self.model.post_quant_conv.parameters()), lr = lr, eps = 1e-8, betas = (args.beta1, args.beta2)
      ) 
    opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr = lr, eps = 1e-8, betas = (args.beta1, args.beta2))
    
    return opt_vq, opt_disc
  
  def _compute_losses(self, images, decoded_images, q_loss, epoch, accumulation_steps, adapt_disc_factor=True):
      
    # with self.DeviceManager(self.discriminator):
    disc_real = self.discriminator(images)
    disc_fake = self.discriminator(decoded_images)
      
    disc_factor = self.args.disc_factor
    if adapt_disc_factor:
      disc_factor = self.model.adapt_weight(self.args.disc_factor, epoch, self.args.disc_start_epoch)
        
    # with self.DeviceManager(self.perceptual_loss):
    perceptual_loss = self.perceptual_loss(images, decoded_images)

    lpips = perceptual_loss.mean()
    # fid = self.fid(images, decoded_images, type="tensor")  ## takes too much memory, evaluate in test phase
    fid = 0.0
  
    rec_loss = torch.abs(images - decoded_images)
    perceptual_rec_loss = self.args.perceptual_loss_factor * lpips + self.args.rec_loss_factor * rec_loss
    perceptual_rec_loss = perceptual_rec_loss.mean()
    g_loss = -torch.mean(disc_fake)
        
    ## TODO: Reconcile Q-Loss and KL Divergence loss  
    # with self.DeviceManager(self.model):
    lmbda = self.model.calculate_lambda(perceptual_rec_loss, g_loss)
    
    vq_loss = (perceptual_rec_loss + q_loss + disc_factor * lmbda * g_loss) / accumulation_steps
        
    d_loss_real = torch.mean(F.relu(1. - disc_real))
    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
    gan_loss = (disc_factor * 0.5 * (d_loss_real + d_loss_fake)) / accumulation_steps
    
    return vq_loss, gan_loss, lpips, fid
  
  def train(self):
    args = self.args
    dir_name = setup_directories(args.run_name)
    save_yaml_file(args, dir_name)
    
    accumulation_steps = args.gradient_accumulation_steps

    train_dataloader, val_dataloader, train_data, val_data, data = load_dataset(args,dataset="CIFAR10",with_captions=False, split=True)
    
    fabric = Fabric(accelerator=args.device, devices=1, precision="bf16-mixed") ## precision="32"
    fabric.launch()
    
    with fabric.init_module():
      self.model = VQGAN(args)
      self.discriminator = Discriminator(args)
      self.discriminator.apply(autoencoder_weights_init)
      
    self.perceptual_loss = LPIPS().eval().to(device=args.device)
    # self.fid = FID(device=args.device)
    self.opt_vq, self.opt_disc = self.configure_optimizers(args)

    self.model, self.opt_vq = fabric.setup(self.model, self.opt_vq)
    self.discriminator, self.opt_disc = fabric.setup(self.discriminator, self.opt_disc)
    
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    if args.load_pretrained_weights:
      if args.vqgan_checkpoint_path != 'None':
        self.model.load_checkpoint(args.vqgan_checkpoint_path)
      if args.vqgan_optimizer_path != 'None':
        self.opt_vq.load_state_dict(torch.load(args.vqgan_optimizer_path))
      if args.discriminator_checkpoint_path != 'None':
        self.discriminator.load_checkpoint(args.discriminator_checkpoint_path)
      if args.vqgan_optimizer_path != 'None':
        self.opt_disc.load_state_dict(torch.load(args.discriminator_optimizer_path))
    
    logger = SummaryWriter(os.path.join("runs", args.run_name))

    best_val_loss = None
    best_train_loss = None

    # vq_cycle = True  
    
    for epoch in range(args.epochs):
      
      epoch_train_loss = 0.0
      
      with tqdm(range(len(train_dataloader))) as pbar1:
        for i, (images,_) in zip(pbar1, train_dataloader):
          
          # with self.DeviceManager(self.model):
          decoded_images, _, q_loss = self.model(images)

          vq_loss, gan_loss, lpips, fid = self._compute_losses(images, decoded_images, q_loss, epoch, accumulation_steps)
          
          fabric.backward(vq_loss,retain_graph=True)
          fabric.backward(gan_loss)
          # Consider training generator and discrimantor periodically in 'n' iterations rather than in same iteration
          # if (epoch+1) % args.alternative_cycle_epochs == 0:
          #   vq_cycle = not vq_cycle
          # if vq_cycle:
          #   fabric.backward(vq_loss,retain_graph=True)
          # else:
          #   fabric.backward(gan_loss)

          epoch_train_loss += (vq_loss.item() + gan_loss.item()) * len(images)
          
          if (i+1) % accumulation_steps == 0:
            self.opt_vq.step()
            self.opt_disc.step()
            self.opt_vq.zero_grad()
            self.opt_disc.zero_grad()
            
          if (i+1) % args.image_save_step_interval == 0:
            with torch.no_grad():
              real_fake_images = torch.cat((images.add(1).mul(0.5), decoded_images.add(1).mul(0.5)))
              vutils.save_image(real_fake_images, os.path.join("results", args.run_name, dir_name, f"{epoch}_{i}_train.jpg"), nrow=images.shape[0])
          
          pbar1.set_postfix(VQ_Loss=np.round(vq_loss.detach().cpu().numpy(), 5), GAN_Loss=np.round(gan_loss.detach().cpu().numpy(), 3),
                           BatchLPIPS=np.round(lpips.detach().cpu().numpy(),5),BatchFID=np.round(fid,5))
          pbar1.update(0)
          logger.add_scalar("Loss/train/vq-loss", vq_loss.item(), global_step=(epoch*len(pbar1) + i))
          logger.add_scalar("Loss/train/gan-loss", gan_loss.item(), global_step=(epoch*len(pbar1) + i))
          logger.add_scalar("Metric/train/batch-lpips", lpips.item(), global_step=(epoch*len(pbar1) + i))
          logger.add_scalar("Metric/train/batch-fid", fid, global_step=(epoch*len(pbar1) + i))

      epoch_train_loss /= len(train_data)

      if best_train_loss is None or epoch_train_loss < best_train_loss:
        best_train_loss = epoch_train_loss
        self.save_model(args, dir_name, "best_train_loss")

      if (epoch+1) % args.image_save_epoch_interval == 0:
        
        logging.info(f"Evaluating validation set at epoch {epoch}:")
        pbar2 = tqdm(val_dataloader)

        query_index = random.randint(0, len(pbar2) - 1)
        epoch_val_loss = 0.0

        for j, (val_images, _) in enumerate(pbar2):
          
          # with self.DeviceManager(self.model):
          decoded_val_images, _, q_loss_val = self.model(val_images)

          vq_loss_val, gan_loss_val, lpips_val, fid_val = self._compute_losses(val_images, decoded_val_images, q_loss_val, epoch, 
                                                                               accumulation_steps, adapt_disc_factor=False)

          pbar2.set_postfix(VQ_Loss=np.round(vq_loss_val.detach().cpu().numpy(), 5), GAN_Loss=np.round(gan_loss_val.detach().cpu().numpy(), 3),
                          BatchLPIPS=np.round(lpips_val.detach().cpu().numpy(),5), BatchFID=np.round(fid_val,5))
          pbar2.update(0)
          logger.add_scalar("Loss/valid/vq-loss", vq_loss_val.item(), global_step=(epoch//args.image_save_epoch_interval)*len(pbar2) + j)
          logger.add_scalar("Loss/valid/gan-loss", gan_loss_val.item(), global_step=(epoch//args.image_save_epoch_interval)*len(pbar2) + j)
          logger.add_scalar("Metric/valid/batch-lpips", lpips_val.item(), global_step=(epoch//args.image_save_epoch_interval)*len(pbar2) + j)
          logger.add_scalar("Metric/valid/batch-fid", fid_val, global_step=(epoch//args.image_save_epoch_interval)*len(pbar2) + j)

          epoch_val_loss += (vq_loss_val.item() + gan_loss_val.item()) * len(val_images)
          
          if j == query_index:
            with torch.no_grad():
              real_fake_val_images = torch.cat((val_images.add(1).mul(0.5), decoded_val_images.add(1).mul(0.5)))
              vutils.save_image(real_fake_val_images, os.path.join("results", args.run_name, dir_name, f"{epoch}_{j}_valid.jpg"), nrow=val_images.shape[0])

        epoch_val_loss /= len(val_data)
        
        if best_val_loss is None or epoch_val_loss < best_val_loss:
          best_val_loss = epoch_val_loss
          self.save_model(args, dir_name, "best_val_loss")

      self.save_model(args, dir_name, "latest")

  def save_model(self, args, dir_name, name):
      torch.save(self.model.state_dict(), os.path.join("models", args.run_name, dir_name, f"vqgan_{name}_ckpt.pt"))
      torch.save(self.opt_vq.state_dict(), os.path.join("models", args.run_name, dir_name, f"vqgan_{name}_optim.pt"))
      torch.save(self.discriminator.state_dict(), os.path.join("models", args.run_name, dir_name, f"discriminator_{name}_ckpt.pt"))
      torch.save(self.opt_disc.state_dict(), os.path.join("models", args.run_name, dir_name, f"discriminator_{name}_optim.pt"))

class DDPMTrainer:
  
  def __init__(self, args):
    self.args = args
    self.model = None
    self.ema_model = None
    self.diffusion = None
    self.use_cfg = None
    self.use_clip = None
    self.clip = None
    self.mse = None
  
  def train(self):

    args = self.args
    dir_name = setup_directories(args.run_name)
    save_yaml_file(args, dir_name)

    accumulation_steps = args.gradient_accumulation_steps
    
    dataloader, data = load_dataset(args,dataset="CIFAR10", with_captions=True, split=False)
    
    fabric = Fabric(accelerator=args.device, devices=1, precision="32")
    fabric.launch()
    
    with fabric.init_module():
      # self.model = UNet(num_classes=args.num_classes, with_context=True)
      self.model = UNet(num_classes=args.num_classes, with_context=True,
                   context_dim=args.context_dim, time_dim=args.time_dim, input_size=args.image_size)
      if args.use_clip:
        self.clip = CLIPModel(args)
        self.clip.load_checkpoint(args.clip_checkpoint_path)
        self.clip.eval().requires_grad_(False)
    
    self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate)
    
    self.model, self.optimizer = fabric.setup(self.model, self.optimizer)

    dataloader = fabric.setup_dataloaders(dataloader)
    
    self.mse = nn.MSELoss()
    self.diffusion = Diffusion(img_size=args.image_size, device=args.device, noise_steps=args.denoise_steps)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

    image_save_steps = [int(i * args.image_save_subepoch_interval * l) 
                          for i in range(1, int(1/args.image_save_subepoch_interval))]
    
    self.optimizer.zero_grad()

    for epoch in range(args.epochs):
      logging.info(f"Starting epoch {epoch}:")
      pbar = tqdm(dataloader)

      for i, (images, annotations) in enumerate(pbar):
        t = self.diffusion.sample_timestamps(images.shape[0]).to(args.device)
        x_t, noise = self.diffusion.noise_images(images, t)
        
        labels = annotations['class']
        captions = annotations['caption']

        n = len(captions)
        
        if args.use_cfg and np.random.random() < args.label_noise:
          labels = None
        
        context = None
        if args.use_clip:
          context, _ = self.clip.encode_texts(captions)

        predicted_noise = self.model(x_t, t, context, labels)

        loss = self.mse(noise, predicted_noise) / accumulation_steps

        fabric.backward(loss)

        if (i+1) % accumulation_steps == 0:
          self.optimizer.step() 
          ema.step_ema(self.ema_model, self.model)
          self.optimizer.zero_grad() 
        
        pbar.set_postfix(MSE=loss.item())
        logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
        
        if (i+1) in image_save_steps: 
          self.evaluate(args, dir_name, epoch, i, captions, labels)

      if (epoch+1) % args.image_save_epoch_interval == 0: 
          self.evaluate(args, dir_name, epoch, len(pbar)-1, captions, labels)
          
      torch.save(self.model.state_dict(), os.path.join("models", args.run_name, dir_name, f"ckpt.pt"))
      torch.save(self.ema_model.state_dict(), os.path.join("models", args.run_name, dir_name, f"ckpt_ema.pt"))
      torch.save(self.optimizer.state_dict(), os.path.join("models", args.run_name, dir_name, f"optim.pt"))

  def evaluate(self, args, dir_name, epoch, i, captions, labels):

    context = None
    if args.use_clip:
      context, _ = self.clip.encode_texts(captions)
      if args.use_cfg:
        uncond_context, _ = self.clip.encode_texts(["" for _ in range(len(captions))])
        context = torch.cat([context, uncond_context])

    sampled_images = self.diffusion.sample(self.model, n=len(captions), context=context, labels=labels, cfg_scale=args.cfg_scale, )
    ema_sampled_images = self.diffusion.sample(self.ema_model, n=len(captions), context=context, labels=labels, cfg_scale=args.cfg_scale)
    ddpm_save_images(sampled_images, os.path.join("results", args.run_name, dir_name, f"{epoch}_{i}_sampled.jpg"))
    ddpm_save_images(ema_sampled_images, os.path.join("results", args.run_name, dir_name, f"{epoch}_{i}_ema_sampled.jpg"))

class SDTrainer:
  
  def __init__(self, args):
    self.args = args
    self.vqgan = None
    self.clip = None
    self.model = None  ## The UNET model to be trained
    self.ema_model = None
    self.diffusion = None
    self.discriminator = None
    self.perceptual_loss = None
    self.opt_sd = None
    self.opt_vq = None
    self.opt_disc = None
    self.mse = None
    self.fid = None
    
  def train(self):
    args = self.args
    dir_name = setup_directories(args.run_name)
    save_yaml_file(args, dir_name)

    accumulation_steps = args.gradient_accumulation_steps
    
    train_dataloader, val_dataloader, train_data, val_data, data = load_dataset(args,dataset="CIFAR10", with_captions=True, split=True)
    
    fabric = Fabric(accelerator=args.device, devices=1, precision="32")  ## Experiment with half precision bf16-mixed, 
    fabric.launch()                                                      ## which most likely won't produce NaNs unlike mixed float16'  
    
    with fabric.init_module():
      self.model = UNet(in_ch=args.latent_channels, out_ch=args.latent_channels, num_classes=args.num_classes, context_dim=args.context_dim, time_dim=args.time_dim,
                        arch_downscaling=args.vqgan_encoder_downscaling, with_context=True)
      self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
      self.vqgan = VQGAN(args)
      self.vqgan.load_checkpoint(args.vqgan_checkpoint_path)
      self.vqgan.eval().requires_grad_(False)
      self.perceptual_loss = LPIPS().eval().to(device=args.device)
      self.diffusion = Diffusion(img_size=None, img_channels=None, noise_steps=args.denoise_steps, device=args.device)
      self.clip = CLIPModel(args)
      self.clip.load_checkpoint(args.clip_checkpoint_path)
      if not args.train_conditioning_model:
        self.clip.eval().requires_grad_(False)

    if args.train_conditioning_model:
      self.opt_sd = optim.AdamW(list(self.model.parameters())+list(self.clip.parameters()), 
                                lr=args.learning_rate) 
    else:
      self.opt_sd = optim.AdamW(self.model.parameters(), lr=args.learning_rate)
    
    self.model, self.opt_sd = fabric.setup(self.model, self.opt_sd)
    
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    self.mse = nn.MSELoss()
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(train_dataloader)
    ema = EMA(args.ema_decay)

    encoded_image_shape = None
    best_val_loss = None
    best_train_loss = None

    self.opt_sd.zero_grad()

    for epoch in range(args.epochs):
      logging.info(f"Starting epoch {epoch}:")
      pbar1 = tqdm(train_dataloader)
      
      epoch_train_loss = 0.0

      for i, (images, annotations) in enumerate(pbar1):
        
        labels = annotations['class']
        captions = annotations['caption']
        
        encoded_images, _, _ = self.vqgan.encode(images)

        if encoded_image_shape is None:
          encoded_image_shape = encoded_images.shape[1:]
        
        t = self.diffusion.sample_timestamps(encoded_images.shape[0]).to(args.device)
        x_t, noise = self.diffusion.noise_images(encoded_images, t)

        if not args.use_labels or np.random.random() < args.label_noise:
          labels = None
        
        context, _ = self.clip.encode_texts(captions)

        predicted_noise = self.model(x_t, t, context, labels)
        loss = self.mse(noise, predicted_noise) / accumulation_steps

        fabric.backward(loss)

        epoch_train_loss += loss * len(images)

        if (i+1) % accumulation_steps == 0:
          self.opt_sd.step() 
          ema.step_ema(self.ema_model, self.model)
          self.opt_sd.zero_grad() 

        if (i+1) % args.image_save_step_interval == 0:
          with torch.no_grad():
            sampled_decoded_images, sampled_decoded_images_ema = self.sample_images(args.use_cfg, args.cfg_scale, len(captions), 
                                                                                    encoded_image_shape, labels, captions)
      
            # real_fake_images = torch.cat((images.add(1).mul(0.5), 
            # sampled_decoded_images.add(1).mul(0.5), sampled_decoded_images_ema.add(1).mul(0.5)), dim=0)
            real_fake_images = torch.cat((normalize_images(images), 
                                          normalize_images(sampled_decoded_images), 
                                          normalize_images(sampled_decoded_images_ema)), dim=0)
            vutils.save_image(real_fake_images, os.path.join("results", args.run_name, dir_name, f"{epoch}_{i}_train.jpg"), nrow=len(captions))
        
        pbar1.set_postfix(MSE=loss.item())
        logger.add_scalar("Loss/train/mse", loss.item(), global_step=epoch * l + i)
        
      epoch_train_loss /= len(train_data)

      if best_train_loss is None or epoch_train_loss < best_train_loss:
        best_train_loss = epoch_train_loss
        self.save_model(args, dir_name, "best_train_loss")

      if (epoch+1) % args.image_save_epoch_interval == 0:

        logging.info(f"Evaluating validation set at epoch {epoch}:")
        pbar2 = tqdm(val_dataloader)

        query_index = random.randint(0, len(pbar2) - 1)
        epoch_val_loss = 0.0
        
        with torch.no_grad():
          for j, (val_images, val_annotations) in enumerate(pbar2):
            valid_labels = val_annotations['class']
            valid_captions = val_annotations['caption']

            val_encoded_images, _, _ = self.vqgan.encode(val_images)
          
            val_t = self.diffusion.sample_timestamps(val_encoded_images.shape[0]).to(args.device)
            val_x_t, val_noise = self.diffusion.noise_images(val_encoded_images, val_t)
            
            if not args.use_labels:
              valid_labels = None

            valid_context, _ = self.clip.encode_texts(valid_captions)

            val_predicted_noise = self.model(val_x_t, val_t, valid_context, valid_labels)
            val_loss = self.mse(val_noise, val_predicted_noise) / accumulation_steps

            epoch_val_loss += val_loss * len(val_images)

            if j == query_index:
              with torch.no_grad():
                sampled_decoded_images_val, sampled_decoded_images_ema_val = self.sample_images(args.use_cfg, args.cfg_scale, len(valid_captions), 
                                                                                                encoded_image_shape, valid_labels, valid_captions)
              
                # real_fake_images = torch.cat((val_images.add(1).mul(0.5), 
                # sampled_decoded_images_val.add(1).mul(0.5), sampled_decoded_images_ema_val.add(1).mul(0.5)), dim=0)
                real_fake_images = torch.cat((normalize_images(val_images), 
                                              normalize_images(sampled_decoded_images_val), 
                                              normalize_images(sampled_decoded_images_ema_val)), dim=0)
                vutils.save_image(real_fake_images, os.path.join("results", args.run_name, dir_name, f"{epoch}_{j}_valid.jpg"), nrow=len(valid_captions))

            pbar2.set_postfix(MSE=val_loss.item())
            logger.add_scalar("Loss/valid/mse", val_loss.item(), global_step=(epoch//args.image_save_epoch_interval)*len(pbar2) + j)
        
        epoch_val_loss /= len(val_data)
        
        if best_val_loss is None or epoch_val_loss < best_val_loss:
          best_val_loss = epoch_val_loss
          self.save_model(args, dir_name, "best_val_loss")

      self.save_model(args, dir_name, "latest")

  def save_model(self, args, dir_name, name):
      
      torch.save(self.model.state_dict(), os.path.join("models", args.run_name, dir_name, f"sd_{name}_ckpt.pt"))
      torch.save(self.ema_model.state_dict(), os.path.join("models", args.run_name, dir_name, f"sd_{name}_ckpt_ema.pt"))
      torch.save(self.opt_sd.state_dict(), os.path.join("models", args.run_name, dir_name, f"sd_{name}_optim.pt"))
      
      if args.train_conditioning_model:
        torch.save(self.clip.state_dict(), os.path.join("models", args.run_name, dir_name, f"clip_{name}_ckpt.pt"))

  def sample_images(self, use_cfg, cfg_scale, num_samples, encoded_image_shape, labels, captions, normalize=False):
      
      context, _ = self.clip.encode_texts(captions)
      if use_cfg:
        uncond_context, _ = self.clip.encode_texts(["" for _ in range(len(captions))])
        context = torch.cat([context, uncond_context])
      
      sampled_encoded_images = self.diffusion.sample(self.model, n=num_samples, 
                                                          context=context, labels=labels, 
                                                          image_shape=encoded_image_shape,
                                                          cfg_scale=cfg_scale,
                                                          normalize_output=False)
      sampled_encoded_images_ema = self.diffusion.sample(self.ema_model, n=num_samples, 
                                                              context=context, labels=labels, 
                                                              image_shape=encoded_image_shape,
                                                              cfg_scale=cfg_scale,
                                                              normalize_output=False)
          
      sampled_decoded_images = self.vqgan.decode(sampled_encoded_images)
      sampled_decoded_images_ema = self.vqgan.decode(sampled_encoded_images_ema)

      if normalize:
        sampled_decoded_images = normalize_images(sampled_decoded_images)
        sampled_decoded_images_ema = normalize_images(sampled_decoded_images_ema)

      return sampled_decoded_images,sampled_decoded_images_ema

class CLIPTrainer:
  def __init__(self, args, loss = None):
    self.args = args
    self.model = None
    self.optimizer = None
    self.temperature = args.temperature
    self.loss_function = clip_loss

  def _compute_loss(self, image_projections, text_projections, accumulation_steps):
    similarity = text_projections @ image_projections.T
    loss = self.loss_function(similarity / self.temperature)
    img_accuracy, cap_accuracy = clip_metrics(similarity)
    return loss / accumulation_steps, img_accuracy, cap_accuracy
  
  def train(self):
    args = self.args
    dir_name = setup_directories(args.run_name)
    save_yaml_file(args, dir_name)

    accumulation_steps = args.gradient_accumulation_steps
    
    train_dataloader, val_dataloader, train_data, val_data, data = load_dataset(args,dataset="CIFAR10", 
                                                                with_captions=True, split=True, shuffle_valid=False)
    
    fabric = Fabric(accelerator=args.device, devices=1, precision="32")
    fabric.launch()
    
    with fabric.init_module():
      self.model = CLIPModel(args)
    
    self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate)
    
    if args.load_pretrained_weights:
        if args.clip_checkpoint_path is not None:
          self.model.load_checkpoint(args.clip_checkpoint_path)
        if args.clip_optimizer_path is not None:
          self.optimizer.load_state_dict(torch.load(args.clip_optimizer_path))

    self.model, self.optimizer = fabric.setup(self.model, self.optimizer)

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(train_dataloader)
    
    best_val_loss = None
    best_train_loss = None

    for epoch in range(args.epochs):
      logging.info(f"Starting epoch {epoch}:")
      pbar1 = tqdm(train_dataloader)
      
      epoch_train_loss = 0.0

      for i, (images, annotations) in enumerate(pbar1):
        labels = annotations['class']
        captions = annotations['caption']

        image_projections, text_projections = self.model.query(images, captions)
        loss, img_acc, cap_acc  = self._compute_loss(image_projections, text_projections, accumulation_steps)
        
        fabric.backward(loss)

        if (i+1) % accumulation_steps == 0:
          self.optimizer.step()
          self.optimizer.zero_grad()  
        
        pbar1.set_postfix({'LogSoftmax':loss.item(), 'ImageAccuracy':img_acc.item(),'CaptionAccuracy':cap_acc.item()})
        logger.add_scalar("Loss/train", loss.item(), global_step=epoch * len(pbar1) + i)

        epoch_train_loss += loss.item()*len(images)

      epoch_train_loss /= len(train_data)

      if best_train_loss is None or epoch_train_loss < best_train_loss:
        best_train_loss = epoch_train_loss
        torch.save(self.model.state_dict(), os.path.join("models", args.run_name, dir_name, f"clip_best_train_loss_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", args.run_name, dir_name, f"clip_best_train_loss_optim.pt"))

      if (epoch+1) % args.image_save_epoch_interval == 0:

        logging.info(f"Evaluating validation set at epoch {epoch}:")

        pbar2 = tqdm(val_dataloader)

        all_valid_image_projections = []
        all_valid_image_indices = []

        ## pick a random image and caption batch from the validation set to use as query
        query_index = random.randint(0, len(pbar2) - 1)

        epoch_val_loss = 0.0

        with torch.no_grad():
          for j, (valid_images, valid_annotations) in enumerate(pbar2):

            valid_captions = valid_annotations['caption']
            valid_indices = valid_annotations['index']
            valid_image_projections, valid_text_projections = self.model.query(valid_images, valid_captions)

            all_valid_image_projections.append(valid_image_projections)
            all_valid_image_indices.append(valid_indices)
            valid_loss, valid_img_acc, valid_cap_acc = self._compute_loss(valid_image_projections, valid_text_projections, accumulation_steps)

            pbar2.set_postfix({'LogSoftmax':valid_loss.item(), 'ImageAccuracy':valid_img_acc.item(),'CaptionAccuracy':valid_cap_acc.item()})
            logger.add_scalar("Loss/valid", valid_loss.item(), global_step=(epoch//args.image_save_epoch_interval)*len(pbar2) + j)
            epoch_val_loss += valid_loss.item()*len(valid_images)
            
            if j == query_index:
              query_annotations = { k:v.detach().clone() if isinstance(v, torch.Tensor) else v for k,v in valid_annotations.items() }
        
        epoch_val_loss /= len(val_data)

        if best_val_loss is None or epoch_val_loss < best_val_loss:
          best_val_loss = epoch_val_loss
          torch.save(self.model.state_dict(), os.path.join("models", args.run_name, dir_name, f"clip_best_val_loss_ckpt.pt"))
          torch.save(self.optimizer.state_dict(), os.path.join("models", args.run_name, dir_name, f"clip_best_val_loss_optim.pt")) 
        
        all_valid_image_projections = torch.cat(all_valid_image_projections, dim=0)
        all_valid_image_indices = torch.cat(all_valid_image_indices, dim=0)

        with torch.no_grad():
          match_indices = self.model.find_matches(query_annotations['caption'], all_valid_image_projections, 
                                                  max_matches=args.max_matches, num_captions=1)

          retrieval_table = []
          for k in range(len(match_indices)):
            retrieval_table.append({
              'query_filepath': data.samples[(query_annotations['index'][k]).item()][0],
              'query_caption': query_annotations['caption'][k],
              'retrieved_filepaths': [data.samples[all_valid_image_indices[j].item()][0] for j in match_indices[k]], 
              'retrieved_captions': [data[all_valid_image_indices[j].item()][1]['caption'] for j in match_indices[k]]
            })

          clip_plot_images(retrieval_table, save_plot=True, show_figure=False,
                           save_path=os.path.join("results", args.run_name, dir_name, f"{epoch}_{query_index}_valid.jpg"), 
                           max_matches=args.max_matches)

      torch.save(self.model.state_dict(), os.path.join("models", args.run_name, dir_name, f"clip_latest_ckpt.pt"))
      torch.save(self.optimizer.state_dict(), os.path.join("models", args.run_name, dir_name, f"clip_latest_optim.pt"))

def launch():
  config_parser = ArgumentParser()
  config_parser.add_argument("--config", help="Path to config file")
  config_args = config_parser.parse_args()
  
  with open(config_args.config, "r") as yaml_file:
    config = yaml.safe_load(yaml_file)
    for key,value in config.items():
      config_parser.add_argument(f"--{key}", type=type(value), default=value)
      
  args = config_parser.parse_args()
  
  if args.run_name == "DDPMConditional":
    trainer = DDPMTrainer(args)
  elif args.run_name == "VQGAN":
    trainer = VQGANTrainer(args)  
  elif args.run_name == "SD":
    trainer = SDTrainer(args)  
  elif args.run_name == "CLIP":
    trainer = CLIPTrainer(args)  

  trainer.train()

if __name__ == '__main__':
  launch()
  
