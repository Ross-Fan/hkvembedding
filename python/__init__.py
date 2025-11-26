"""
HKV Python Bindings
High-performance GPU hashtable for embeddings using HierarchicalKV
"""

# 导入编译好的核心模块
try:
    import hkv_core
except ImportError as e:
    raise ImportError(
        f"Failed to import hkv_core module: {e}\n"
        "Make sure the package is properly installed and CUDA is available."
    )

# 导入Python层的封装
from .hkvembedding import HierarchicalHashEmbedding
from .utils import *

__version__ = "1.0.0"
__author__ = "HKV Team"

# 导出核心类和函数
__all__ = [
    "HierarchicalHashEmbedding",
    "hash_ids",
    # 从hkv_core导出的类
    "HashTable", 
    "Int64HashTable",
    "version",
]

# 重新导出hkv_core的类，方便使用
HashTable = hkv_core.HashTable
Int64HashTable = hkv_core.Int64HashTable
version = hkv_core.version

# 便捷创建函数
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
