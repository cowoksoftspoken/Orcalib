import orca.nn as nn
from orca.tensor import Tensor
from typing import Optional
import numpy as np

class BERTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.TransformerBlock(embed_dim, num_heads, dim_feedforward, dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.block(x, mask)

class BERT(nn.Module):
    """
    BERT model architecture blueprint.
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_size = hidden_size
        
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.ln = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        
        # Stack of blocks
        layers = []
        for _ in range(num_hidden_layers):
            layers.append(BERTBlock(hidden_size, num_attention_heads, 4 * hidden_size, dropout))
        self.encoder = nn.Sequential(*layers)
        
        # Pooler
        self.pooler_dense = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(self, x: Tensor, token_type_ids: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        # x is one-hot token tensor of shape [batch, seq_len, vocab_size]
        batch, seq_len, _ = x.shape
        
        # Generate one-hot position tensor
        pos_np = np.zeros((batch, seq_len, self.max_position_embeddings), dtype=np.float32)
        for i in range(seq_len):
            pos_np[:, i, i % self.max_position_embeddings] = 1.0
        dev = "gpu" if "gpu" in str(x.device) else "cpu"
        pos = Tensor.from_list(pos_np.flatten().tolist(), shape=[batch, seq_len, self.max_position_embeddings]).to(dev)
        
        embeddings = self.word_embeddings(x) + self.position_embeddings(pos)
        
        if token_type_ids is not None:
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        else:
            zeros_np = np.zeros((batch, seq_len, self.type_vocab_size), dtype=np.float32)
            zeros_np[:, :, 0] = 1.0
            zeros = Tensor.from_list(zeros_np.flatten().tolist(), shape=[batch, seq_len, self.type_vocab_size]).to(dev)
            embeddings = embeddings + self.token_type_embeddings(zeros)
            
        embeddings = self.ln(embeddings)
        embeddings = self.drop(embeddings)
        
        h = embeddings
        for block in self.encoder._modules.values():
            h = block(h, mask=mask)
            
        return h
