import torch
import torch.nn as nn
from flash_attn.modules.mha import FlashSelfAttention
import math

from config import ModelConfig

def rope_rotate(x: torch.Tensor):
    # For each pair of indices 2i, 2i+1
    # theta_i = 10000^(-2i/d)
    # x'[2i] = x[2i]cos(position * theta_i) - x[2i + 1]sin(position * theta_i)
    # x'[2i+1] = x[2i]sin(position * theta_i) + x[2i + 1]cos(position * theta_i)
    batch_size, seq_len, embedding_dim = x.shape
    i = torch.arange(embedding_dim // 2, device=x.device)
    inverse_frequency = 1.0 / (10000 ** (i / (embedding_dim // 2)))
    position_indices = torch.arange(seq_len, device=x.device)
    rotation_angles = torch.outer(position_indices, inverse_frequency)
    sin_angles = rotation_angles.sin()
    cos_angles = rotation_angles.cos()

    first_half = x[..., :embedding_dim // 2]
    second_half = x[..., embedding_dim // 2:]
    rotated_embeddings = torch.cat([
        first_half * cos_angles - second_half * sin_angles,
        first_half * sin_angles + second_half * cos_angles
    ], dim=1)
    return rotated_embeddings

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, causal=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.flash_attention = FlashSelfAttention(causal=True, attention_dropout=0.1)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim)
        )
        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.norm2 = nn.LayerNorm(self.embedding_dim)
        self._init_params()

    def _init_params(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))
                if mod.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(mod.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(mod.bias, -bound, bound)
            elif isinstance(mod, nn.LayerNorm):
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, x):
        attn_out, _ = self.flash_attention(x)
        x = self.norm1(attn_out)
        x = self.feed_forward(x)
        x = self.norm2(x)
        return x

class LLM(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.config = model_config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                causal=self.config.causal
            ) for _ in range(self.config.depth)
        ])
        self.final_norm = nn.LayerNorm(self.config.embedding_dim)
        self.output = nn.Linear(self.config.embedding_dim, self.config.vocab_size, bias=False)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor):
        x = self.embedding(tokens)
        x = rope_rotate(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.output(x)
        return logits        

