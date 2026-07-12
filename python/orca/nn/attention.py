import orca
import math
from typing import Optional
from .module import Module
from .linear import Linear
from .dropout import Dropout
from .activation import Softmax

class MultiHeadAttention(Module):
    """
    Allows the model to jointly attend to information from different representation
    subspaces as described in the paper: Attention Is All You Need.
    
    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads. Note that `embed_dim` must be divisible by `num_heads`.
        dropout (float, optional): Dropout probability on the attention weights. Default: 0.0.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.q_proj = Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)
        
        self.dropout = Dropout(dropout)
        self.softmax = Softmax(dim=-1)

    def forward(self, query: orca.Tensor, key: orca.Tensor, value: orca.Tensor, attn_mask: Optional[orca.Tensor] = None) -> orca.Tensor:
        """
        Forward pass for MultiHeadAttention.
        
        Args:
            query (Tensor): Query tensor of shape `(batch_size, seq_len, embed_dim)`.
            key (Tensor): Key tensor of shape `(batch_size, seq_len, embed_dim)`.
            value (Tensor): Value tensor of shape `(batch_size, seq_len, embed_dim)`.
            attn_mask (Optional[Tensor], optional): Attention mask. Used for causal or padding masks.
                Contains 0.0 for unmasked and -1e9 for masked positions.
                
        Returns:
            Tensor: Output tensor of shape `(batch_size, seq_len, embed_dim)`.
        """
        # query, key, value shape: [batch_size, seq_len, embed_dim]
        batch_size = query.shape[0]
        q_len = query.shape[1]
        k_len = key.shape[1]
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        # [batch_size, seq_len, num_heads, head_dim]
        q = q.reshape([batch_size, q_len, self.num_heads, self.head_dim])
        k = k.reshape([batch_size, k_len, self.num_heads, self.head_dim])
        v = v.reshape([batch_size, k_len, self.num_heads, self.head_dim])
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores: Q @ K^T / sqrt(head_dim)
        # q: [b, h, q_len, d]
        # k: [b, h, k_len, d]
        # output: [b, h, q_len, k_len]
        attn_weights = orca.einsum("bhqd,bhkd->bhqk", q, k)
        
        # Scale
        scale_factor = 1.0 / math.sqrt(self.head_dim)
        attn_weights = attn_weights * scale_factor
        
        if attn_mask is not None:
            # Mask should be broadcastable to [batch_size, num_heads, q_len, k_len]
            attn_weights = attn_weights + attn_mask
            
        # Numerically stable softmax
        attn_probs = self.softmax(attn_weights)
        
        attn_probs = self.dropout(attn_probs)
        
        # Attention output: attn_probs @ V
        # attn_probs: [b, h, q_len, k_len]
        # v: [b, h, k_len, d]
        # output: [b, h, q_len, d]
        attn_output = orca.einsum("bhqk,bhkd->bhqd", attn_probs, v)
        
        # Transpose back and reshape to [batch_size, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape([batch_size, q_len, self.embed_dim])
        
        # Final projection
        output = self.out_proj(attn_output)
        return output
