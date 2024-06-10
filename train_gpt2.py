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

class MLP(nn.Module):
    def __init__(self, config):
        super().__innit__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # 
        self.gelu = nn.GELU(approximate='tanh') # no real benefit nowadays compared to exact # more modern models are now using 'SwiGLU', 'RoPE' and ohers
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
        self.attn = CausalSelfAtention(config)
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