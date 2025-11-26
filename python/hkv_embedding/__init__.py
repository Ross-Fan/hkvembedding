"""
HKV Embedding
High-performance GPU hashtable embeddings for PyTorch using HierarchicalKV
"""

# 导入编译好的核心模块
try:
    from .hkv_core import HashTable, Int64HashTable, version
except ImportError as e:
    raise ImportError(
        f"Failed to import hkv_core module: {e}\n"
        "Make sure the package is properly installed and CUDA is available.\n"
        "Try running: pip install --force-reinstall hkv-embedding"
    )

# 导入Python层的封装
from .hkvembedding import HierarchicalHashEmbedding
from .utils import *
from .optimizer import HKVOptimizer, HKVAdamOptimizer

__version__ = "1.0.0"
__author__ = "HKV Team"

__all__ = [
    "HierarchicalHashEmbedding",
    "HashTable", 
    "Int64HashTable",
    "version",
    "hash_ids",
    "create_hashtable",
    "create_embedding",
    "HKVOptimizer",
    "HKVAdamOptimizer",
]

def create_hashtable(init_capacity, max_capacity, embedding_dim, max_hbm_gb=16, key_type="uint64"):
    """
    创建HashTable的便捷函数
    
    Args:
        init_capacity: 初始容量
        max_capacity: 最大容量
        embedding_dim: 嵌入维度
        max_hbm_gb: 最大HBM使用量(GB)
        key_type: 键类型 ("uint64" 或 "int64")
    
    Returns:
        HashTable实例
    """
    if key_type == "uint64":
        return HashTable(init_capacity, max_capacity, embedding_dim, max_hbm_gb)
    elif key_type == "int64":
        return Int64HashTable(init_capacity, max_capacity, embedding_dim, max_hbm_gb)
    else:
        raise ValueError(f"Unsupported key_type: {key_type}")

def create_embedding(embedding_dim, max_capacity=1000000, init_capacity=100000, 
                    max_hbm_gb=16, device='cuda', dtype=None, init_std=None):
    """
    创建HierarchicalHashEmbedding的便捷函数
    
    Args:
        embedding_dim: 嵌入向量维度
        max_capacity: 最大容量
        init_capacity: 初始容量
        max_hbm_gb: 最大GPU内存使用量(GB)
        device: 设备类型
        dtype: 数据类型
        init_std: 初始化标准差
    
    Returns:
        HierarchicalHashEmbedding实例
    """
    return HierarchicalHashEmbedding(
        embedding_dim=embedding_dim,
        max_capacity=max_capacity,
        init_capacity=init_capacity,
        max_hbm_gb=max_hbm_gb,
        device=device,
        dtype=dtype,
        init_std=init_std
    )

# 版本信息
def get_version_info():
    """获取版本信息"""
    try:
        core_version = version()
    except:
        core_version = "unknown"
    
    return {
        "package_version": __version__,
        "core_version": core_version,
        "author": __author__
    }
