import math

import torch
from torch import nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, head_size, num_embed, block_size):
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5

        wei = wei.masked_fill(self.trill[:T,:T]== 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v=self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.module):
    def __init__(self, num_heads,head_size,num_embed, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    head_size=head_size,
                    num_embed=num_embed,
                    block_size=block_size,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj == nn.Linear(num_embed, num_embed)
    

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    
    def __init__(self, num_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embed, 4 * num_embed),
            nn.ReLU(),
            nn.Linear(4* num_embed, num_embed)
        )

    def forward(self, x):
        return self.net(x)