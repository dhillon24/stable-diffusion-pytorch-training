import os
import torch
import requests
import timm
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
from scipy import linalg
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights
from torchvision import transforms
from collections import namedtuple

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

def download(url, local_path, chunk_size=1024):
  os.makedirs(os.path.split(local_path)[0], exist_ok=True)
  with requests.get(url, stream=True) as r:
    total_size = int(r.headers.get("content-length", 0))
    with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
      with open(local_path, "wb") as f:
        for data in r.iter_content(chunk_size=chunk_size):
          if data:
            f.write(data)
            pbar.update(chunk_size)
            

def get_cktpt_path(name, root):
  assert name in URL_MAP
  path = os.path.join(root, CKPT_MAP[name])
  if not os.path.exists(path):
    print(f"Downloading {name} from {URL_MAP[name]} to {path}")
    download(URL_MAP[name], path)
  return path

class ScalingLayer(nn.Module):
  def __init__(self):
    super().__init__()
    self.register_buffer("shift", torch.Tensor([-0.30, -0.88, -0.188])[None, :, None, None])
    self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])
    
  def forward(self, x):
    return (x - self.shift) / self.scale
  
class NetLinLayer(nn.Module):
  def __init__(self, in_ch, out_ch=1):
    super().__init__()
    self.model = nn.Sequential(
      nn.Dropout(),
      nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
    )
    
  def forward(self):
    raise NotImplementedError()
    
class VGG16(nn.Module):
  def __init__(self):
    super().__init__()
    vgg_pretrained_features = vgg16(weights=VGG16_Weights.DEFAULT).features
    # weights=VGG16_Weights.IMAGENET1K_V1 can be used above for previous pretrained=True behavior
    slices = [vgg_pretrained_features[i] for i in range(30)]
    self.slice1 = nn.Sequential(*slices[0:4])
    self.slice2 = nn.Sequential(*slices[4:9])
    self.slice3 = nn.Sequential(*slices[9:16])
    self.slice4 = nn.Sequential(*slices[16:23])
    self.slice5 = nn.Sequential(*slices[23:30])
    
    for param in self.parameters():
      param.requires_grad = False
      
  def forward(self, x):
    h = self.slice1(x)
    h_relu1 = h
    h = self.slice2(h)
    h_relu2 = h
    h = self.slice3(h)
    h_relu3 = h
    h = self.slice4(h)
    h_relu4 = h
    h = self.slice5(h)
    h_relu5 = h
    vgg_outputs = namedtuple("VGGOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
    return vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

class LPIPS(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.scaling_layer = ScalingLayer()
    self.ch = [64, 128, 256, 512, 512]
    self.vgg = VGG16()
    self.lins = nn.ModuleList([NetLinLayer(self.ch[i]) for i in range(len(self.ch))])
    
    self.load_from_pretrained()
    
    for param in self.parameters():
      param.requires_grad = False
      
  def load_from_pretrained(self, name="vgg_lpips"):
    ckpt = get_cktpt_path(name, "vgg_lpips")
    self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
    
  @staticmethod  
  def __norm_tensor(x):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + 1e-10)
  
  @staticmethod
  def __spatial_average(x):
    return x.mean([2,3], keepdim=True)
  
  def compute_activations(self, images):
    return self.vgg(self.scaling_layer(images))
  
  def compute_similarity(self, features_real, features_fake):
    diffs = {}
    for i in range(len(self.ch)):
      diffs[i] = (self.__norm_tensor(features_real[i]) - self.__norm_tensor(features_fake[i])) ** 2
    return sum([self.__spatial_average(self.lins[i].model(diffs[i])) for i in range(len(self.ch))])  
    
  def forward(self, real_x, fake_x):
    features_real = self.compute_activations(real_x)
    features_fake = self.compute_activations(fake_x)
    return self.compute_similarity(features_real, features_fake)
  

class FID:
  
  def __init__(self, device):
    super().__init__()
    self.inception = timm.create_model('inception_v3', 
                                       pretrained=True, num_classes=0, global_pool="avg").to(device)
    self.inception.eval()
    self.input_size = 299
    self.resizer = transforms.Resize((self.input_size, self.input_size), antialias=True)
    for param in self.inception.parameters():
      param.requires_grad = False

  def compute_activations(self, images):
    return self.inception(self.resizer(images))
  
  def compute_distance(self, features_real, features_fake, eps=1e-6, type="numpy"):

    if type == "tensor":
      features_real = features_real.detach().cpu().numpy()
      features_fake = features_fake.detach().cpu().numpy()
    
    mu1 = np.atleast_1d(features_real.mean(axis=0))
    sigma1 = np.atleast_2d(np.cov(features_real, rowvar=False))
    mu2 = np.atleast_1d(features_fake.mean(axis=0))
    sigma2 = np.atleast_2d(np.cov(features_fake, rowvar=False))

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all(): ## account for singular product
      msg = ('fid calculation produces singular product; '
             'adding %s to diagonal of cov estimates') % eps
      logging.warn(msg)
      offset = np.eye(sigma1.shape[0]) * eps
      covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean): ## negligible img part generated due to numerical error
      if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        m = np.max(np.abs(covmean.imag))
        raise ValueError('Imaginary component {}'.format(m))
      covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
  
  def __call__(self, real_x, fake_x, eps=1e-6, type="numpy"):
    features_real = self.compute_activations(real_x)
    features_fake = self.compute_activations(fake_x)
    return self.compute_distance(features_real, features_fake, eps, type)

  
