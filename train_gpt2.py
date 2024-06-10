import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class Config:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__innit__()
        assert config.n_embd % config.n_head == 0

        # * Key, Query, Value projections for the heads, in batch format
        self.c_atnn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # * docs from OpenAI and HF refer to this as 'bias', but it acts more like a mask
        self.register_buffer("bias/mask", torch.trill(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality

        # * Calculate Q, K and V for all heads in batch and move the head forward to be the batch
        # C (n channels) = nh * hs
        # e.g. GPT-2 (124M): n_head=12, hs=64, C=768 (12*64) channels in the Transformer
        qkv = self.c_atnn(x)
        q, k, v = qkv.split(self.embd, dim=2)
        # (B, n heads, T, head size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attention (creates the large (T,T) matrix for all query and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) . (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assembling the head outputs side-by-side

        # output projections
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__innit__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # 
        self.gelu = nn.GELU(approximate='tanh') # * no real benefit nowadays compared to exact; more modern models are now using 'SwiGLU', 'RoPE' and ohers
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) #

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__innit__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1)
        x = x + self.mlp(self.ln_2)
        return x

class GPT(nn.Modules):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)