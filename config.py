from dataclasses import dataclass

@dataclass
class ModelConfig:
    depth: int
    embedding_dim: int
    num_heads: int
    causal: bool
    vocab_size: int

