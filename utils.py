import os
import torch
import random
import logging
import torchvision
import yaml

import numpy as np
import albumentations as alb
import torch.nn as nn

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

def setup_directories(run_name):
  os.makedirs("models", exist_ok=True)
  os.makedirs("results", exist_ok=True)
  os.makedirs(os.path.join("models", run_name), exist_ok=True)
  os.makedirs(os.path.join("results", run_name), exist_ok=True)
  
  dir_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
  os.makedirs(os.path.join("models", run_name, dir_name), exist_ok=True)
  os.makedirs(os.path.join("results", run_name, dir_name), exist_ok=True)
  
  return dir_name

def save_yaml_file(args, dir_name):
  with open(os.path.join("models", args.run_name, dir_name, "config.yaml"), "w") as f:
    yaml.dump(args, f)

def normalize_images(x, quantize=False):
  x = (x.clamp(-1,1) + 1) / 2
  x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=1.0)
  if quantize:
    x = (x * 255).type(torch.uint8)
  return x

class DeviceContextManager():
    def __init__(self, module, active_device='gpu', idle_device='cpu'):
        self.module = module
        self.active_device = active_device
        self.idle_device = idle_device
         
    def __enter__(self):
        self.module.to(self.active_device)
        return self.module
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.module.to(self.idle_device)
 
## DDPM Utils ##

def ddpm_plot_images(images):
  plt.figure(figsize=(32, 32))
  plt.imshow(torch.cat([torch.cat([i for i in images.cpu()], dim=-1)], dim=-2).permute(1,2,0).cpu())
  plt.show()
  
def ddpm_save_images(images, path, **kwargs):
  grid = torchvision.utils.make_grid(images, **kwargs)
  ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
  im = Image.fromarray(ndarr)
  im.save(path)
  
def ddpm_get_data(args):
  transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(args.image_size + args.image_size//4),
    torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
  dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)#, num_workers=4, pin_memory=True)
  return dataloader
  
## Autoencoder utils ##
  
class CIFARImagePaths(Dataset):
  
  class_prefix = "class"
  
  def __init__(self, path, size=None):
    self.size = size
    self.path = path
    self.samples = [(os.path.join(dirpath, name), dirpath.split(os.sep)[-1]) for dirpath, _, files in os.walk(path) 
               for name in files if name.endswith((".jpg", ".png", ".bmp"))]
    random.shuffle(self.samples)
    self.length = len(self.samples)
    self.rescaler = alb.SmallestMaxSize(max_size=self.size)
    self.cropper = alb.CenterCrop(height=self.size, width=self.size)
    self.normalizer = alb.Normalize()
    self.preprocessor = alb.Compose([self.rescaler, self.cropper, self.normalizer])
    # self.preprocessor = alb.Compose([self.rescaler, self.cropper])
    
  def __len__(self):
    return self.length
    
  def __getitem__(self,i):
    example = self._preprocess_image(self.samples[i][0])
    label = self.samples[i][1]
    return example, self.class_to_index(label)
  
  def _preprocess_image(self, image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
      image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = self.preprocessor(image=image)['image']
    # image = (image / 127.5 - 1.0).astype(np.float32)  ## Normalize to [0,1] when not using albumentations Normalize standardization
    image = image.astype(np.float32)
    image = image.transpose((2, 0, 1))
    return image

  def class_to_index(self, label):
    return int(label[len(self.class_prefix):])
  
class CIFARImagePathsWithCaptions(CIFARImagePaths):
    
    def __init__(self, path, captions_file, size=None):
      super().__init__(path, size)
      self.captions = {}
      with open(captions_file, "r") as f:
        for line in f.readlines():
          line = line.strip().split(",")
          self.captions[line[0]] = line[1]

    def __len__(self):
      return self.length

    def __getitem__(self,i):
      image = super()._preprocess_image(self.samples[i][0])
      label = self.samples[i][1]
      filepath = self.samples[i][0]
      filename = filepath[filepath.find(self.path)+len(self.path)+1:]
      return image, {'class': super().class_to_index(label), 'caption': self.captions[filename], 'index': i}
    
  
def load_dataset(args, dataset="CIFAR10", with_captions=False, split=False, train_val_split=0.9, shuffle=True, shuffle_valid=True):
  if dataset == "CIFAR10":
    if with_captions:
      data = CIFARImagePathsWithCaptions(args.dataset_path, args.captions_file, size=args.image_size)
    else:
      data = CIFARImagePaths(args.dataset_path, size=args.image_size)
    if not split:
      return DataLoader(data, batch_size=args.batch_size, shuffle=shuffle), data
    train_data, val_data = torch.utils.data.random_split(data, [int(len(data) * train_val_split), len(data) - int(len(data) * train_val_split)])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=shuffle)#, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=shuffle_valid)#, num_workers=4, pin_memory=True)
    return train_loader, val_loader, train_data, val_data, data
  return None

def autoencoder_weights_init(module):
  classname = module.__class__.__name__
  if classname.find("Conv")!= -1:
    nn.init.normal_(module.weight.data, 0.0, 0.02)
  elif classname.find("BatchNorm")!= -1:
    nn.init.normal_(module.weight.data, 1.0, 0.02)
    nn.init.constant_(module.bias.data, 0.0)
    
def autoencoder_plot_images(images):
  x = images['input']
  reconstruction = images['rec']
  half_sample = images['half_sample']
  full_sample = images['full_sample']
  
  fig, axarr = plt.subplots(1, 4)
  axarr[0].imshow(x.cpu().detach().numpy()[0].tranpose(1, 2, 0))
  axarr[1].imshow(reconstruction.detach().numpy()[0].tranpose(1, 2, 0))
  axarr[2].imshow(half_sample.cpu().detach().numpy()[0].tranpose(1, 2, 0))
  axarr[3].imshow(full_sample.cpu().detach().numpy()[0].tranpose(1, 2, 0))
  plt.show()
  
def output_free_memory(msg = "", logged = None):
  if logged is None:
    logged = logging
  logging.info(f"{msg}")
  tb = torch.cuda.get_device_properties(0).total_memory
  rb = torch.cuda.memory_reserved(0)
  ab = torch.cuda.memory_allocated(0)
  t = tb / (1024 ** 3)
  r = rb / (1024 ** 3)
  a = ab / (1024 ** 3)
  f = (rb-ab) / (1024  ** 3) #free inside reserved
  logged.info(f"Free memory now: {f:.2f} GB, Memory Allocated: {a:.2f} GB, Memory Reserved: {r:.2f} GB, Total Memory: {t:.2f} GB")


  ## CLIP Utils ##

def clip_plot_images(retrieval_table, save_plot=False, show_figure=False, save_path=None, max_matches=5, query_image_available=True):

  num_rows = len(retrieval_table)
  num_cols = max_matches + 1
  fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols*4, num_rows*4))

  ax2d = np.atleast_2d(ax)

  for i in range(num_rows):
    query_image = Image.open(retrieval_table[i]['query_filepath']) if query_image_available else Image.new('RGB',(64,64),"rgb(255,0,255)")
    query_caption = retrieval_table[i]['query_caption']

    retrieved_images = [Image.open(image_filepath) for image_filepath 
                        in retrieval_table[i]["retrieved_filepaths"][:max_matches]]
    retrieved_captions = [caption for caption in retrieval_table[i]["retrieved_captions"][:max_matches]]

    ax2d[i, 0].imshow(query_image)
    ax2d[i, 0].set_title(query_caption)

    for j in range(max_matches):
      ax2d[i, j+1].imshow(retrieved_images[j])
      ax2d[i, j+1].set_title(retrieved_captions[j])

    for j in range(num_cols):
      ax2d[i, j].axis('off')

  if save_plot:
    plt.savefig(save_path)

  if show_figure:
    plt.show()
    
  # plt.close('all')

 