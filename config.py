from dataclasses import dataclass

@dataclass
class ModelConfig:
    depth: int
    embedding_dim: int
    num_heads: int
    causal: bool
    vocab_size: int

small_LLM = ModelConfig(
    depth=4,
    embedding_dim=768,
    num_heads=4,
    causal=True,
    vocab_size=pass
)

