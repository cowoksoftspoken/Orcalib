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
        # x is assumed to be a one-hot encoded tensor of shape (batch, sequence_length, num_embeddings)
        # We need to flatten to 2D for matmul: (batch * sequence_length, num_embeddings)
        original_shape = x.shape
        batch = original_shape[0]
        seq_len = original_shape[1]
        
        x_flat = x.reshape([batch * seq_len, self.num_embeddings])
        out_flat = x_flat @ self.weight.tensor
        
        # Reshape back to (batch, sequence_length, embedding_dim)
        return out_flat.reshape([batch, seq_len, self.embedding_dim])
