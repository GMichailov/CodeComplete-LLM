import torch
import torch.nn as nn
from flash_attn.modules.mha import FlashSelfAttention

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

class Transformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_latents=4, causal=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.flash_attention = FlashSelfAttention(causal=True, attention_dropout=0.1)
        

