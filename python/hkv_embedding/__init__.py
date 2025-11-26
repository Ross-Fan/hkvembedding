"""
HKV Embedding
High-performance GPU hashtable embeddings for PyTorch using HierarchicalKV

Designed for large-scale recommendation systems with billions of unique IDs.
Uses NVIDIA's HierarchicalKV for GPU-accelerated hash table storage.
"""

# Import compiled core module
try:
    from .hkv_core import HashTable, Int64HashTable, version
except ImportError as e:
    raise ImportError(
        f"Failed to import hkv_core module: {e}\n"
        "Make sure the package is properly installed and CUDA is available.\n"
        "Try running: pip install --force-reinstall hkv-embedding"
    )

# Import Python layer wrappers
from .hkvembedding import (
    HierarchicalHashEmbedding,
    MultiTableHKVEmbedding,
    GradientBuffer,
    AdamStateBuffer,
    HKVEmbeddingFunction,
)
from .utils import hash_ids
from .optimizer import (
    HKVOptimizer, 
    HKVAdamOptimizer, 
    HKVSparseAdam,
    HKVAdagrad,
)

__version__ = "1.1.0"
__author__ = "HKV Team"

__all__ = [
    # Core embedding classes
    "HierarchicalHashEmbedding",
    "MultiTableHKVEmbedding",
    
    # Low-level hash tables
    "HashTable", 
    "Int64HashTable",
    
    # Optimizers
    "HKVOptimizer",
    "HKVAdamOptimizer",
    "HKVSparseAdam",
    "HKVAdagrad",
    
    # Utilities
    "hash_ids",
    "create_hashtable",
    "create_embedding",
    "version",
    
    # Internal (for advanced usage)
    "GradientBuffer",
    "AdamStateBuffer",
    "HKVEmbeddingFunction",
]


def create_hashtable(init_capacity: int, max_capacity: int, embedding_dim: int, 
                     max_hbm_gb: int = 16, key_type: str = "uint64"):
    """
    Create HashTable convenience function.
    
    Args:
        init_capacity: Initial capacity
        max_capacity: Maximum capacity
        embedding_dim: Embedding dimension
        max_hbm_gb: Maximum HBM usage (GB)
        key_type: Key type ("uint64" or "int64")
    
    Returns:
        HashTable instance
    """
    if key_type == "uint64":
        return HashTable(init_capacity, max_capacity, embedding_dim, max_hbm_gb)
    elif key_type == "int64":
        return Int64HashTable(init_capacity, max_capacity, embedding_dim, max_hbm_gb)
    else:
        raise ValueError(f"Unsupported key_type: {key_type}. Use 'uint64' or 'int64'.")


def create_embedding(embedding_dim: int, 
                     max_capacity: int = 1000000, 
                     init_capacity: int = 100000, 
                     max_hbm_gb: int = 16, 
                     device: str = 'cuda', 
                     dtype=None, 
                     init_std: float = None,
                     learning_rate: float = 0.01,
                     weight_decay: float = 0.0):
    """
    Create HierarchicalHashEmbedding convenience function.
    
    Args:
        embedding_dim: Embedding vector dimension
        max_capacity: Maximum capacity
        init_capacity: Initial capacity
        max_hbm_gb: Maximum GPU memory usage (GB)
        device: Device type
        dtype: Data type
        init_std: Initialization standard deviation
        learning_rate: Learning rate for built-in optimizer
        weight_decay: Weight decay
    
    Returns:
        HierarchicalHashEmbedding instance
    """
    import torch
    return HierarchicalHashEmbedding(
        embedding_dim=embedding_dim,
        max_capacity=max_capacity,
        init_capacity=init_capacity,
        max_hbm_gb=max_hbm_gb,
        device=device,
        dtype=dtype or torch.float32,
        init_std=init_std,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )


def get_version_info() -> dict:
    """Get version information."""
    try:
        core_version = version()
    except:
        core_version = "unknown"
    
    return {
        "package_version": __version__,
        "core_version": core_version,
        "author": __author__
    }
