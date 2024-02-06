import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.in_ch = in_ch
    self.out_ch = out_ch
    self.block = nn.Sequential(
      nn.GroupNorm(32, in_ch),
      nn.SiLU(),
      nn.Conv2d(in_ch, out_ch, 3, 1, 1),
      nn.GroupNorm(32, out_ch),
      nn.SiLU(),
      nn.Conv2d(out_ch, out_ch, 3, 1, 1),
    )
    
    if in_ch != out_ch:
      self.channel_up = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
      
  def forward(self, x):
    if self.in_ch != self.out_ch:
      return self.channel_up(x) + self.block(x)
    else:
      return x + self.block(x)
      
class NonLocalBlock(nn.Module):
  def __init__(self, in_ch):
    super().__init__()
    self.in_ch = in_ch
    self.gn = nn.GroupNorm(32, in_ch)
    self.q = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
    self.k = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
    self.v = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
    self.proj_out = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
    
  def forward(self, x):
    h_ = self.gn(x)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    
    b, c, h, w = q.shape
    q = q.reshape(b, c, h*w)
    q = q.permute(0, 2, 1)
    k = k.reshape(b, c, h*w)
    v = v.reshape(b, c, h*w)
    
    attn = torch.bmm(q, k)
    attn = attn * (int(c)**(-0.5))
    attn = F.softmax(attn, dim=2)
    attn = attn.permute(0, 2, 1)
    
    A = torch.bmm(v, attn)
    A = A.reshape(b, c, h, w)
    
    return x + A
    
class UpSampleBlock(nn.Module):
  def __init__(self, ch):
    super().__init__()
    self.conv = nn.Conv2d(ch, ch, 3, 1, 1)
  
  def forward(self, x):
    x = F.interpolate(x, scale_factor=2.0)
    return self.conv(x)
  
class DownSampleBlock(nn.Module):
  def __init__(self, ch):
    super().__init__()
    self.conv = nn.Conv2d(ch, ch, 3, 2, 0)
    self.pad = (0, 1, 0, 1)
  
  def forward(self, x):
    x = F.pad(x, self.pad, mode="constant", value=0)
    return self.conv(x)

class Encoder(nn.Module):
  def __init__(self, args):
    super().__init__()
    ch = [128, 128, 128, 256, 256, 512]  # large
    # ch = [128, 256, 256, 512] # small 16x
    attn_resolutions = [16]
    num_res_blocks = 2
    resolution = 256 #64
    layers = [nn.Conv2d(args.image_channels, ch[0], 3, 1, 1)]

    for i in range(len(ch)-1):
      in_ch = ch[i]
      out_ch = ch[i+1]
      for j in range(num_res_blocks):
        layers.append(ResidualBlock(in_ch, out_ch))
        in_ch = out_ch
        if resolution in attn_resolutions:
          layers.append(NonLocalBlock(in_ch))
      if i != len(ch) - 2:
        layers.append(DownSampleBlock(ch[i+1]))
        resolution // 2
    layers.append(ResidualBlock(ch[-1], ch[-1]))
    layers.append(NonLocalBlock(ch[-1]))
    layers.append(ResidualBlock(ch[-1], ch[-1]))
    layers.append(nn.GroupNorm(32, ch[-1]))
    layers.append(nn.SiLU())
    layers.append(nn.Conv2d(ch[-1], args.latent_channels, 3, 1, 1))
    self.model = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.model(x)
  
  
class Decoder(nn.Module):
  def __init__(self, args):
    super().__init__()
    ch = [512, 256, 256, 128, 128] # large
    # ch = [512, 256, 128] # small
    attn_resolutions = [16]
    num_res_blocks = 3
    resolution = 16
    
    in_ch = ch[0]
    layers = [nn.Conv2d(args.latent_channels, in_ch, 3, 1, 1),
              ResidualBlock(in_ch, in_ch),
              NonLocalBlock(in_ch),
              ResidualBlock(in_ch, in_ch)]
              
    for i in range(len(ch)):
      out_ch = ch[i]
      for j in range(num_res_blocks):
        layers.append(ResidualBlock(in_ch, out_ch))
        in_ch = out_ch
        if resolution in attn_resolutions:
          layers.append(NonLocalBlock(in_ch))
      if i != 0:
        layers.append(UpSampleBlock(in_ch))
        resolution *= 2
          
    layers.append(nn.GroupNorm(32, in_ch))
    layers.append(nn.SiLU())
    layers.append(nn.Conv2d(in_ch, args.image_channels, 3, 1, 1))
    self.model = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.model(x)
  
  
class Codebook(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.num_codebook_vectors = args.num_codebook_vectors
    self.latent_dim = args.latent_dim
    self.beta = args.beta
    
    self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
    self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)
    
  def forward(self, z):
    z = z.permute(0, 2, 3, 1).contiguous()
    z_flattened = z.view(-1, self.latent_dim)
    
    d = torch.sum(z_flattened**2, dim=1, keepdim=True) +\
        torch.sum(self.embedding.weight**2, dim=1) -\
          2*(torch.matmul(z_flattened, self.embedding.weight.t()))
    
    min_encoding_indices = torch.argmin(d, dim=1)
    z_q = self.embedding(min_encoding_indices).view(z.shape)
    
    loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
    z_q = z + (z_q - z).detach()
    z_q = z_q.permute(0, 3, 1, 2)
    
    return z_q, min_encoding_indices, loss
    
  
class VQGAN(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.encoder = Encoder(args)
    self.decoder = Decoder(args)
    self.codebook = Codebook(args)
    self.quant_conv = nn.Conv2d(args.latent_channels, args.latent_channels, 1)
    self.post_quant_conv = nn.Conv2d(args.latent_channels, args.latent_channels, 1) 

  def encode(self, x):
    x = self.encoder(x)
    x = self.quant_conv(x)
    codebook_mapping, codebook_indices, q_loss = self.codebook(x)
    return codebook_mapping, codebook_indices, q_loss
  
  def decode(self, z):
    z = self.post_quant_conv(z)
    z = self.decoder(z) 
    return z    
    
  def forward(self, x):
    x = self.encoder(x)
    x = self.quant_conv(x)
    x, codebook_indices, q_loss = self.codebook(x)
    x = self.post_quant_conv(x)
    x = self.decoder(x)
    return x, codebook_indices, q_loss
  
  def calculate_lambda(self, perceptual_loss, gan_loss):
    last_layer = self.decoder.model[-1]
    last_layer_weight = last_layer.weight
    perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
    gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]
    
    lmbda = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
    lmbda = torch.clamp(lmbda, 0, 1e4).detach()
    return 0.8 * lmbda

  @staticmethod
  def adapt_weight(disc_factor, i, threshold, value=0.):
    if i < threshold:
      disc_factor = value
    return disc_factor
  
  def load_checkpoint(self, path):
    self.load_state_dict(torch.load(path))
      

class Discriminator(nn.Module):
  def __init__(self, args, num_filters_last=64, n_layers=3):
    super().__init__()
    
    layers = [nn.Conv2d(args.image_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
    num_filters_mult = 1
    
    for i in range(1, n_layers + 1):
      num_filters_mult_last = num_filters_mult
      num_filters_mult = min(2 ** i, 8)
      
      layers += [
        nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                  2 if i < n_layers else 1, 1, bias=False),
        nn.BatchNorm2d(num_filters_last * num_filters_mult),
        nn.LeakyReLU(0.2, True)
      ]
    
    layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
    self.model = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.model(x)
  
  def load_checkpoint(self, path):
    self.load_state_dict(torch.load(path))
             

  
    
