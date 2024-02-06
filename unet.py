
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention
from collections import OrderedDict

class DoubleConv(nn.Module):
  def __init__(self, in_ch, out_ch, mid_ch=None, residual=False):
    super().__init__()
    self.residual = residual
    if not mid_ch:
      mid_ch = out_ch
    self.double_conv = nn.Sequential(
      nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
      nn.GroupNorm(1, mid_ch),
      nn.GELU(),
      nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
      nn.GroupNorm(1, out_ch),
    )
      
  def forward(self, x):
    if self.residual:
      return F.gelu(x + self.double_conv(x))
    else:
      return self.double_conv(x)
      
class Down(nn.Module):
  def __init__(self, in_ch, out_ch, embed_dim=256):
    super().__init__()
    self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
                                       DoubleConv(in_ch, in_ch, residual=True),
                                       DoubleConv(in_ch, out_ch))
    
    self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, out_ch))
    
  def forward(self, x, t):
    x = self.maxpool_conv(x)
    emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
    return x + emb
  
class Up(nn.Module):
  def __init__(self, in_ch, out_ch, embed_dim=256):
    super().__init__()
    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.conv = nn.Sequential(DoubleConv(in_ch, in_ch, residual=True),
                               DoubleConv(in_ch, out_ch, in_ch//2))
    
    self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, out_ch))
    
  def forward(self, x, skip_x, t):
    x = self.up(x)
    x = torch.cat([skip_x, x], dim=1)
    x = self.conv(x)
    emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
    return x + emb
  
class Projection2D(nn.Module):
  def __init__(self, in_ch, in_size, out_size):
    super().__init__()
    self.in_ch = in_ch
    self.in_size = in_size
    self.out_size = out_size
    self.proj = nn.Linear(in_size ** 2, out_size ** 2)
    
  def forward(self, x):
    initial_shape = x.shape
    x = x.view(initial_shape[0], self.in_ch, self.in_size ** 2)
    x = self.proj(x)
    return x.view(initial_shape[0], self.in_ch, self.out_size, self.out_size)
     
class SelfAttentionBlock(nn.Module):
  def __init__(self, in_ch, size):
    super().__init__()
    self.in_ch = in_ch
    self.size = size
    self.mha = nn.MultiheadAttention(in_ch, 4, batch_first=True)
    # self.mha = SelfAttention(4, in_ch, in_proj_bias=False)  
    self.ln = nn.LayerNorm([in_ch])
    self.ff_self = nn.Sequential(nn.LayerNorm([in_ch]),
                                 nn.Linear(in_ch, in_ch), 
                                 nn.GELU(), 
                                 nn.Linear(in_ch, in_ch))
    
  def forward(self, x):
    # x = x.view(-1, self.in_ch, self.size * self.size).swapaxes(1, 2)
    batch_size = x.shape[0]
    x = x.view(batch_size, self.in_ch, -1).swapaxes(2,1)
    x_ln = self.ln(x)
    attention_value, _ = self.mha(x_ln, x_ln, x_ln)
    # attention_value = self.mha(x_ln)
    attention_value = attention_value + x
    attention_value = self.ff_self(attention_value) + attention_value
    # return attention_value.swapaxes(2,1).view(-1, self.in_ch, self.size, self.size)
    return attention_value.swapaxes(2,1).view(batch_size, self.in_ch, self.size, self.size)
  
class CrossAttentionBlock(nn.Module):
  def __init__(self, in_ch, size, d_context):
    super().__init__()
    self.in_ch = in_ch
    self.size = size
    self.proj = nn.Linear(d_context, in_ch)
    self.mha = nn.MultiheadAttention(in_ch, 4, batch_first=True)
    # self.mha = CrossAttention(4, in_ch, d_context, in_proj_bias=False)
    self.ln = nn.LayerNorm([in_ch])
    self.ff_self = nn.Sequential(nn.LayerNorm([in_ch]),
                                 nn.Linear(in_ch, in_ch), 
                                 nn.GELU(), 
                                 nn.Linear(in_ch, in_ch))
    
  def forward(self, x, y):
    # x = x.view(-1, self.in_ch, self.size * self.size).swapaxes(1, 2)
    batch_size = x.shape[0]
    x = x.view(batch_size, self.in_ch, -1).swapaxes(1, 2)
    x_ln = self.ln(x)
    y_proj = self.proj(y)
    attention_value, _ = self.mha(x_ln, y_proj, y_proj)
    # attention_value = self.mha(x_ln, y)
    attention_value = attention_value + x
    attention_value = self.ff_self(attention_value) + attention_value
    # return attention_value.swapaxes(2,1).view(-1, self.in_ch, self.size, self.size)
    return attention_value.swapaxes(2,1).view(batch_size, self.in_ch, self.size, self.size)

class UNETConditionalScalable(nn.Module):

  def __init__(self, in_ch=3, out_ch=3, time_dim=256, context_dim=768, num_classes=None, device="cuda", with_context=False,
              arch_downscaling=1, input_size=256, bottleneck_size=16):
    
    super().__init__()
    self.device = device
    self.time_dim = time_dim
    self.context_dim = context_dim
    self.arch_downscaling = arch_downscaling
    
    self.module_in_ch = in_ch
    self.module_out_ch = out_ch
    self.with_context = with_context
    
    self.layers = OrderedDict()
    self.layers["inc"] = DoubleConv(self.module_in_ch, 64)
    self.encoder_channels = [(self.module_in_ch, 64)]
    
    in_ch = 64
    out_ch = 128
    max_ch = 256
    res = input_size
    assert bottleneck_size <= input_size and input_size % bottleneck_size == 0
    self.num_levels = int(math.log2(input_size // bottleneck_size))
    attn_res = [64 // (2 ** x) for x in range(int(math.log2(bottleneck_size))-1)]  ## [64, 32, 16]
    for i in range(1, self.num_levels+1):
      self.layers[f"down{i}"] = Down(in_ch, out_ch, embed_dim=time_dim)
      res = res // 2
      if res in attn_res:
        self.layers[f"sa{i}"] = SelfAttentionBlock(out_ch, res // self.arch_downscaling)
        if self.with_context:
          self.layers[f"ca{i}"] = CrossAttentionBlock(out_ch, res // self.arch_downscaling, context_dim)
      self.encoder_channels.append((in_ch, out_ch))
      in_ch = out_ch
      out_ch = min(out_ch * 2, max_ch)
      
    self.layers["bot1"] = DoubleConv(256, 512)
    self.layers["bot2"] = DoubleConv(512, 512)
    self.layers["bot3"] = DoubleConv(512, 256)

    self.decoder_channels = []
    in_ch = 256 + self.encoder_channels[-1][0]
    out_ch = 128
    min_ch = 64
    res = bottleneck_size
    for i in range(1, self.num_levels+1):
      self.layers[f"up{i}"] = Up(in_ch, out_ch, embed_dim=time_dim)
      res *= 2
      if res in attn_res:
        self.layers[f"sa{i+self.num_levels}"] = SelfAttentionBlock(out_ch, res // self.arch_downscaling)
        if self.with_context:
          self.layers[f"ca{i+self.num_levels}"] = CrossAttentionBlock(out_ch, res // self.arch_downscaling, context_dim)
      self.decoder_channels.append((in_ch, out_ch))
      in_ch = out_ch + self.encoder_channels[-i-1][0]
      out_ch = max(out_ch // 2 , min_ch)
    
    self.layers["outc"] = nn.Conv2d(self.decoder_channels[-1][1], self.module_out_ch, kernel_size=1)
    self.decoder_channels.append((self.decoder_channels[-1][1], self.module_out_ch))

    # self.layers = list(self.layers.items())  ## TODO: convert to list of (key, value) pairs for faster inference

    self.num_classes = num_classes
    self.with_context = with_context

    if self.num_classes is not None:
      self.label_emb = nn.Embedding(num_classes, time_dim)
    
  def pos_encoding(self, t, channels):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
    pos_encoding_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    pos_encoding_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    pos_encoding = torch.cat([pos_encoding_a, pos_encoding_b], dim=-1)
    return pos_encoding
  
  def forward(self, x, t, c, y = None):
    t = t.unsqueeze(-1).type(torch.float)
    t = self.pos_encoding(t, self.time_dim)
    
    # TODO: Use spatial transformer to inject additional conditional embeddings like pose into UNET
    
    if self.num_classes is not None and y is not None:
      embedding = self.label_emb(y)
      if t.shape[0] == embedding.shape[0]:
        t += embedding
      else: ## account for case where unconditional input is appended to conditional input
        t += torch.cat([embedding, torch.zeros_like(embedding)], dim=0)
    
    return self._forward(x, t, c)

  def _forward(self, x, t, c):

    outputs = []
    
    x1 = self.layers["inc"](x)

    x = x1
    for i in range(1, self.num_levels+1):
      outputs.append(x)
      x = self.layers[f"down{i}"](x, t)
      if f"sa{i}" in self.layers:
        x = self.layers[f"sa{i}"](x)
        if self.with_context and c is not None:
          x = self.layers[f"ca{i}"](x, c)

    x = self.layers["bot1"](x)
    x = self.layers["bot2"](x)
    x = self.layers["bot3"](x)

    for i in range(1, self.num_levels+1):
      x = self.layers[f"up{i}"](x, outputs[-i], t)
      if f"sa{i+self.num_levels}" in self.layers:
        x = self.layers[f"sa{i+self.num_levels}"](x)
        if self.with_context and c is not None:
          x = self.layers[f"ca{i+self.num_levels}"](x, c)
    
    return self.layers["outc"](x)


class UNetConditional(nn.Module):
  
  def __init__(self, in_ch=3, out_ch=3, time_dim=256, context_dim=768, num_classes=None, device="cuda", with_context=False,
               arch_downscaling=1, input_size=256):
    
    super().__init__()
    self.device = device
    self.time_dim = time_dim
    self.context_dim = context_dim
    self.arch_downscaling = arch_downscaling
    self.projection_required = input_size // 2 > 64
    self.inc= DoubleConv(in_ch, 64)
    self.down1 = Down(64, 128, embed_dim=time_dim)
    if self.projection_required:
      self.proj1 = Projection2D(128, input_size // 2, 32 // self.arch_downscaling)
    self.sa1 = SelfAttentionBlock(128, 32 // self.arch_downscaling)
    self.down2 = Down(128, 256, embed_dim=time_dim)
    self.sa2 = SelfAttentionBlock(256, 16 // self.arch_downscaling)
    self.down3 = Down(256, 256, embed_dim=time_dim)
    self.sa3 = SelfAttentionBlock(256, 8 // self.arch_downscaling)
    
    self.bot1 = DoubleConv(256, 512)
    self.bot2 = DoubleConv(512, 512)
    self.bot3 = DoubleConv(512, 256)

    self.sa4 = SelfAttentionBlock(256, 16 // self.arch_downscaling)
    self.up1 = Up(512, 128, embed_dim=time_dim)
    self.sa5 = SelfAttentionBlock(128, 32 // self.arch_downscaling) 
    self.up2 = Up(256, 64, embed_dim=time_dim)
    self.sa6 = SelfAttentionBlock(64, 64 // self.arch_downscaling)
    if self.projection_required: 
      self.proj2 = Projection2D(64, 64 // self.arch_downscaling, input_size // 2)
    self.up3 = Up(128, 64, embed_dim=time_dim)
    self.outc = nn.Conv2d(64, out_ch, kernel_size=1)
    
    self.num_classes = num_classes
    self.with_context = with_context

    if self.with_context:
      self.ca1 = CrossAttentionBlock(self.sa1.in_ch, self.sa1.size, context_dim)
      self.ca2 = CrossAttentionBlock(self.sa2.in_ch, self.sa2.size, context_dim)
      self.ca3 = CrossAttentionBlock(self.sa3.in_ch, self.sa3.size, context_dim)
      self.ca4 = CrossAttentionBlock(self.sa4.in_ch, self.sa4.size, context_dim)
      self.ca5 = CrossAttentionBlock(self.sa5.in_ch, self.sa5.size, context_dim)
      self.ca6 = CrossAttentionBlock(self.sa6.in_ch, self.sa6.size, context_dim)

    if self.num_classes is not None:
      self.label_emb = nn.Embedding(num_classes, time_dim)
    
  def pos_encoding(self, t, channels):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
    pos_encoding_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    pos_encoding_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    pos_encoding = torch.cat([pos_encoding_a, pos_encoding_b], dim=-1)
    return pos_encoding
  
  def forward(self, x, t, c, y = None):
    t = t.unsqueeze(-1).type(torch.float)
    t = self.pos_encoding(t, self.time_dim)
    
    # TODO: Use spatial transformer to inject additional conditional embeddings like pose into UNET
    
    if self.num_classes is not None and y is not None:
      embedding = self.label_emb(y)
      if t.shape[0] == embedding.shape[0]:
        t += embedding
      else: ## account for case where unconditional input is appended to conditional input
        t += torch.cat([embedding, torch.zeros_like(embedding)], dim=0)
    
    if self.with_context:
      x = self._forward_with_context(x, t, c)
    else:
      x = self._forward(x, t)

    return self.outc(x)
  
  def _forward_with_context(self, x, t, c):

    x1 = self.inc(x)
    x2 = self.down1(x1, t)
    if self.projection_required:
      x2 = self.proj1(x2)
    x2 = self.sa1(x2)
    x2 = self.ca1(x2, c)
    x3 = self.down2(x2, t)
    x3 = self.sa2(x3)
    x3 = self.ca2(x3, c)
    x4 = self.down3(x3, t)
    x4 = self.sa3(x4)
    x4 = self.ca3(x4, c)
    
    x4 = self.bot1(x4)
    x4 = self.bot2(x4)
    x4 = self.bot3(x4)

    x4 = self.sa4(x4)
    x4 = self.ca4(x4, c)
    x = self.up1(x4, x3, t)
    x = self.sa5(x)
    x = self.ca5(x, c)
    x = self.up2(x, x2, t)
    x = self.sa6(x)
    x = self.ca6(x, c)
    if self.projection_required:
      x = self.proj2(x)
    x = self.up3(x, x1, t)

    return x
  
  def _forward(self, x, t):

    x1 = self.inc(x)
    x2 = self.down1(x1, t)
    if self.projection_required:
      x2 = self.proj1(x2)
    x2 = self.sa1(x2)
    x3 = self.down2(x2, t)
    x3 = self.sa2(x3)
    x4 = self.down3(x3, t)
    x4 = self.sa3(x4)
    
    x4 = self.bot1(x4)
    x4 = self.bot2(x4)
    x4 = self.bot3(x4)
    
    x4 = self.sa4(x4)
    x = self.up1(x4, x3, t)
    x = self.sa5(x)
    x = self.up2(x, x2, t)
    x = self.sa6(x)
    if self.projection_required:
      x = self.proj2(x)
    x = self.up3(x, x1, t)

    return x
  
class EMA:
  def __init__(self, beta):
    super().__init__()
    self.beta = beta
    self.step = 0
  
  def update_model_average(self, ema_model, model):
    for current_param, ema_param in zip(model.parameters(), ema_model.parameters()):
      old_weight, new_weight = ema_param.data, current_param.data
      ema_param.data = self.update_average(old_weight, new_weight)
  
  def update_average(self, old, new):
    if old is None:
      return new
    return old * self.beta + (1 - self.beta) * new
    
  def step_ema(self, ema_model, model, step_start_ema=2000):
    if self.step < step_start_ema:
      self.reset_parameters(ema_model, model)
      self.step += 1
      return
    self.update_model_average(ema_model, model)
    self.step += 1
    
  def reset_parameters(self, ema_model, model):
    ema_model.load_state_dict(model.state_dict())   
 
    
  
  



      
      