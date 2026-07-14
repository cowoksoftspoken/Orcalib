from typing import List
from .orca_python import Tensor, Device, DType

def einsum(equation: str, *operands: Tensor) -> Tensor:
    """
    Evaluates the Einstein summation convention on the operands.
    
    Args:
        equation (str): The equation string in Einstein summation convention.
        *operands (Tensor): The tensors to compute the einsum for.
        
    Returns:
        Tensor: The evaluated tensor.
    """
    return Tensor.einsum(equation, list(operands))

def _device_to_str(device) -> str:
    d_str = str(device)
    if "gpu" in d_str or "cuda" in d_str:
        idx = d_str.find("type='")
        if idx != -1:
            idx += 6
            end = d_str.find("'", idx)
            if end != -1:
                return d_str[idx:end]
        return "gpu"
    return "cpu"

def chunk(self, chunks: int, dim: int = -1) -> List[Tensor]:
    """
    Splits the tensor into a specific number of chunks along a given dimension.
    
    Args:
        chunks (int): Number of chunks to split the tensor into.
        dim (int, optional): The dimension along which to split. Default: -1.
        
    Returns:
        List[Tensor]: A list of tensor chunks.
    """
    rank = len(self.shape)
    dim = dim % rank
    
    dim_size = self.shape[dim]
    if dim_size % chunks != 0:
        raise ValueError(f"Dimension size {dim_size} is not divisible by chunks {chunks}")
    chunk_size = dim_size // chunks
    
    t_self = self
    is_rank1 = (rank == 1)
    if is_rank1:
        t_self = self.reshape([1, dim_size])
        rank = 2
        dim = 1
        
    if dim != rank - 1:
        t_self = t_self.transpose(dim, rank - 1)
        
    res_chunks = []
    for i in range(chunks):
        proj_data = [0.0] * (dim_size * chunk_size)
        for j in range(chunk_size):
            row = i * chunk_size + j
            proj_data[row * chunk_size + j] = 1.0
            
        proj_tensor = Tensor.from_list(proj_data, shape=[dim_size, chunk_size]).to(_device_to_str(self.device))
        
        # Match ranks for batched matmul
        if rank > 2:
            target_reshape = [1] * (rank - 2) + [dim_size, chunk_size]
            proj_tensor = proj_tensor.reshape(target_reshape)
            target_expand = list(t_self.shape[:-2]) + [dim_size, chunk_size]
            proj_tensor = proj_tensor.expand(target_expand)
            
        chunk_tensor = t_self @ proj_tensor
        
        if dim != rank - 1:
            chunk_tensor = chunk_tensor.transpose(dim, rank - 1)
            
        if is_rank1:
            chunk_tensor = chunk_tensor.reshape([chunk_size])
            
        res_chunks.append(chunk_tensor)
        
    return res_chunks

# Attach chunk to Tensor class
Tensor.chunk = chunk

__all__ = ["Tensor", "Device", "DType", "einsum"]
