import torch
from torch import nn 
from torch.nn import functional as F
import math

## TODO: Reimplement selattention and crossattention modules. However, note that torch.nn.MultiheadAttention module will have better optimizations

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True,
                 out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):

        input_shape = x.shape
        batch_size, seq_length, d_embed = input_shape
        interim_shape = (batch_size, seq_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        att = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(att, dtype = torch.bool).triu(1)
            att.masked_fill_(mask, -torch.inf)

        att /= math.sqrt(self.d_head)
        att /= F.softmax(att, dim=-1)
        output = att @ v
        output = output.transpose(1,2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)

        return output
    
class CrossAttention(nn.Module):

    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True,
                 out_proj_bias=True):
        super().__init__()
        self.query = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.key = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.value = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        input_shape = x.shape
        batch_size, seq_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.query(x).view(interim_shape).transpose(1,2)
        k = self.key(y).view(interim_shape).transpose(1,2)
        v = self.value(y).view(interim_shape).transpose(1,2)

        att = q @ k.transpose(-1, -2)
        att /= math.sqrt(self.d_head)
        att = F.softmax(att, dim=-1)

        output = att @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output
