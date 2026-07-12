import orca
from typing import Optional
from .module import Module
from .linear import Linear
from .dropout import Dropout
from .normalization import LayerNorm
from .attention import MultiHeadAttention
from .activation import GELU

class TransformerEncoderLayer(Module):
    """
    TransformerEncoderLayer is made up of self-attn and feedforward network.
    
    This standard encoder layer is based on the paper "Attention Is All You Need".
    It uses a Pre-LN (Pre-Layer Normalization) architecture which is typical for 
    more stable training in modern Transformers (like GPT/BERT variants).
    
    Args:
        embed_dim (int): the number of expected features in the input (required).
        num_heads (int): the number of heads in the multiheadattention models (required).
        dim_feedforward (int, optional): the dimension of the feedforward network model. Default: 2048.
        dropout (float, optional): the dropout value. Default: 0.1.
    """
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.linear1 = Linear(embed_dim, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, embed_dim)
        
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        # Simplified GELU for standard transformer MLP
        self.activation = GELU()

    def forward(self, src: orca.Tensor, src_mask: Optional[orca.Tensor] = None) -> orca.Tensor:
        """
        Pass the input through the encoder layer.
        
        Args:
            src (Tensor): the sequence to the encoder layer of shape `(batch, seq_len, embed_dim)`.
            src_mask (Optional[Tensor], optional): the mask for the src sequence. Default: None.
                
        Returns:
            Tensor: Output tensor of shape `(batch, seq_len, embed_dim)`.
        """
        # Pre-LN architecture (typically more stable)
        # Self-attention block
        src_norm = self.norm1(src)
        attn_out = self.self_attn(src_norm, src_norm, src_norm, attn_mask=src_mask)
        src = src + self.dropout1(attn_out)
        
        # MLP block
        src_norm = self.norm2(src)
        mlp_out = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(mlp_out)
        
        return src

class TransformerBlock(TransformerEncoderLayer):
    """
    Alias for `TransformerEncoderLayer`. 
    
    Standard GPT and BERT models are constructed by stacking multiple instances of this block.
    """
    pass

class TransformerDecoderLayer(Module):
    """
    TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    
    Args:
        embed_dim (int): the number of expected features in the input (required).
        num_heads (int): the number of heads in the multiheadattention models (required).
        dim_feedforward (int, optional): the dimension of the feedforward network model. Default: 2048.
        dropout (float, optional): the dropout value. Default: 0.1.
    """
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.multihead_attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.linear1 = Linear(embed_dim, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, embed_dim)
        
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.norm3 = LayerNorm(embed_dim)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        self.activation = GELU()

    def forward(self, tgt: orca.Tensor, memory: orca.Tensor, tgt_mask: Optional[orca.Tensor] = None, memory_mask: Optional[orca.Tensor] = None) -> orca.Tensor:
        """
        Pass the inputs (and mask) through the decoder layer.
        
        Args:
            tgt (Tensor): the sequence to the decoder layer (target).
            memory (Tensor): the sequence from the last layer of the encoder.
            tgt_mask (Optional[Tensor], optional): the mask for the tgt sequence. Default: None.
            memory_mask (Optional[Tensor], optional): the mask for the memory sequence. Default: None.
            
        Returns:
            Tensor: Output tensor.
        """
        # Pre-LN architecture
        tgt_norm = self.norm1(tgt)
        attn_out = self.self_attn(tgt_norm, tgt_norm, tgt_norm, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(attn_out)
        
        tgt_norm = self.norm2(tgt)
        attn_out = self.multihead_attn(tgt_norm, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(attn_out)
        
        tgt_norm = self.norm3(tgt)
        mlp_out = self.linear2(self.dropout(self.activation(self.linear1(tgt_norm))))
        tgt = tgt + self.dropout3(mlp_out)
        
        return tgt
