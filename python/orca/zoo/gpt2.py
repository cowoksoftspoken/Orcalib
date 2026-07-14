import orca.nn as nn
from orca.tensor import Tensor
from typing import Optional
import numpy as np

class GPT2Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.TransformerBlock(embed_dim, num_heads, dim_feedforward, dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.block(x, mask)

class GPT2(nn.Module):
    """
    GPT-2 model architecture blueprint.
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        max_position_embeddings: int = 1024,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.embed_dim = embed_dim
        
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(max_position_embeddings, embed_dim)
        self.drop = nn.Dropout(dropout)
        
        # Stack of blocks
        layers = []
        for _ in range(num_layers):
            layers.append(GPT2Block(embed_dim, num_heads, 4 * embed_dim, dropout))
        self.blocks = nn.Sequential(*layers)
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: Tensor, pos_mask: Optional[Tensor] = None) -> Tensor:
        # x is one-hot token tensor of shape [batch, seq_len, vocab_size]
        batch, seq_len, _ = x.shape
        
        # Generate one-hot position tensor
        pos_np = np.zeros((batch, seq_len, self.max_position_embeddings), dtype=np.float32)
        for i in range(seq_len):
            pos_np[:, i, i % self.max_position_embeddings] = 1.0
        dev = "gpu" if "gpu" in str(x.device) else "cpu"
        pos = Tensor.from_list(pos_np.flatten().tolist(), shape=[batch, seq_len, self.max_position_embeddings]).to(dev)
        
        h = self.wte(x) + self.wpe(pos)
        h = self.drop(h)
        
        for block in self.blocks._modules.values():
            h = block(h, mask=pos_mask)
            
        h = self.ln_f(h)
        original_shape = h.shape
        h_flat = h.reshape([original_shape[0] * original_shape[1], self.embed_dim])
        import orca
        logits_flat = orca.einsum("ij,kj->ik", h_flat, self.wte.weight.tensor)
        logits = logits_flat.reshape([original_shape[0], original_shape[1], self.vocab_size])
        return logits
