import math
from orca import Tensor
from .module import Module
from .parameter import Parameter

class Embedding(Module):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Standard initialization for embeddings
        self.weight = Parameter(Tensor.randn([num_embeddings, embedding_dim], mean=0.0, std=1.0, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        # x is a tensor of indices (typically (batch, sequence_length))
        # The output should be (batch, sequence_length, embedding_dim)
        # Using the native gather method:
        # wait, gather natively takes a dim and an index tensor. 
        # But wait, gather in PyTorch is slightly different from lookup!
        # A lookup table is usually implemented as a specialized operation or via gather if gather supports taking an N-D index and 2-D source.
        # Actually, in our gather implementation: out = src[idx]
        # In our `gather`, the index tensor replaces the `dim` axis.
        # E.g. if weight is (num_embeddings, embedding_dim), and x is (batch, seq_len), 
        # gather(dim=0, index=x) -> (batch, seq_len, embedding_dim)
        
        return self.weight.tensor.gather(0, x)
