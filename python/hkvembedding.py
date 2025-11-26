import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import hkv_core

class HierarchicalHashEmbedding(nn.Module):
    """
    基于HierarchicalKV的PyTorch嵌入层
    
    这是一个高性能的嵌入表实现，支持：
    - 动态容量管理
    - LRU淘汰策略
    - GPU内存优化
    - 自动梯度计算
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 max_capacity: int = 1000000,
                 init_capacity: int = 100000,
                 max_hbm_gb: int = 16,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32,
                 init_std: float = None):
        """
        初始化HierarchicalHashEmbedding
        
        Args:
            embedding_dim: 嵌入向量维度
            max_capacity: 最大容量
            init_capacity: 初始容量
            max_hbm_gb: 最大GPU内存使用量(GB)
            device: 设备类型
            dtype: 数据类型
            init_std: 初始化标准差，默认使用Xavier初始化
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_capacity = max_capacity
        self.init_capacity = init_capacity
        self.device = device
        self.dtype = dtype
        self.init_std = init_std or np.sqrt(2.0 / embedding_dim)
        
        # 创建HKV哈希表
        try:
            self.hashtable = hkv_core.HashTable(
                init_capacity, max_capacity, embedding_dim, max_hbm_gb
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create HashTable: {e}")
        
        # 统计信息
        self.hit_count = 0
        self.miss_count = 0
        self.total_queries = 0
        
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            indices: 输入的特征ID [batch_size] 或 [batch_size, num_features]
        
        Returns:
            embeddings: 嵌入向量
        """
        if indices.numel() == 0:
            return torch.empty(0, self.embedding_dim, dtype=self.dtype, device=self.device)
        
        original_shape = indices.shape
        indices_flat = indices.flatten()
        
        # 查找或插入嵌入
        embeddings = self._find_or_insert_embeddings(indices_flat)
        
        # 恢复原始形状
        if len(original_shape) == 1:
            return embeddings
        else:
            return embeddings.view(*original_shape, self.embedding_dim)
    
    def _find_or_insert_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        """查找或插入嵌入向量"""
        # 转换为numpy数组
        indices_np = indices.cpu().numpy().astype(np.uint64)
        
        try:
            # 调用HKV查找
            embeddings_np, found_flags = self.hashtable.find_or_insert(indices_np)
        except Exception as e:
            raise RuntimeError(f"HKV find_or_insert failed: {e}")
        
        # 转换回PyTorch张量
        embeddings = torch.from_numpy(embeddings_np).to(
            device=self.device, dtype=self.dtype
        )
        
        # 处理新插入的键
        found_tensor = torch.from_numpy(found_flags)
        new_mask = ~found_tensor
        
        if new_mask.any():
            # 随机初始化新嵌入
            new_embeddings = torch.randn(
                new_mask.sum(), self.embedding_dim,
                device=self.device, dtype=self.dtype
            ) * self.init_std
            
            embeddings[new_mask] = new_embeddings
            
            # 更新到哈希表
            self._update_embeddings(indices[new_mask], new_embeddings)
        
        # 更新统计
        self.hit_count += found_flags.sum()
        self.miss_count += len(found_flags) - found_flags.sum()
        self.total_queries += len(found_flags)
        
        return embeddings
    
    def _update_embeddings(self, indices: torch.Tensor, embeddings: torch.Tensor):
        """更新哈希表中的嵌入向量"""
        if indices.numel() == 0:
            return
            
        indices_np = indices.cpu().numpy().astype(np.uint64)
        embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)
        
        try:
            self.hashtable.insert_or_assign(indices_np, embeddings_np)
        except Exception as e:
            raise RuntimeError(f"HKV insert_or_assign failed: {e}")
    
    def lookup(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅查找，不插入新键
        
        Args:
            indices: 查找的键
            
        Returns:
            embeddings: 找到的嵌入向量
            found_mask: 是否找到的掩码
        """
        if indices.numel() == 0:
            return (torch.empty(0, self.embedding_dim, dtype=self.dtype, device=self.device),
                   torch.empty(0, dtype=torch.bool, device=self.device))
        
        indices_flat = indices.flatten()
        indices_np = indices_flat.cpu().numpy().astype(np.uint64)
        
        try:
            embeddings_np, found_flags = self.hashtable.find(indices_np)
        except Exception as e:
            raise RuntimeError(f"HKV find failed: {e}")
        
        embeddings = torch.from_numpy(embeddings_np).to(
            device=self.device, dtype=self.dtype
        )
        found_mask = torch.from_numpy(found_flags).to(self.device)
        
        return embeddings, found_mask
    
    def update(self, indices: torch.Tensor, embeddings: torch.Tensor):
        """
        更新指定键的嵌入向量
        
        Args:
            indices: 要更新的键
            embeddings: 新的嵌入向量
        """
        if indices.numel() == 0:
            return
            
        self._update_embeddings(indices.flatten(), 
                               embeddings.view(-1, self.embedding_dim))
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        try:
            current_size = self.hashtable.size()
            current_capacity = self.hashtable.capacity()
            load_factor = self.hashtable.load_factor()
        except Exception as e:
            current_size = 0
            current_capacity = self.max_capacity
            load_factor = 0.0
        
        hit_rate = self.hit_count / self.total_queries if self.total_queries > 0 else 0
        
        return {
            'current_size': current_size,
            'max_capacity': self.max_capacity,
            'current_capacity': current_capacity,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'total_queries': self.total_queries,
            'hit_rate': hit_rate,
            'load_factor': load_factor,
            'embedding_dim': self.embedding_dim
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.hit_count = 0
        self.miss_count = 0
        self.total_queries = 0
    
    def __repr__(self):
        stats = self.get_statistics()
        return (f"HierarchicalHashEmbedding("
                f"embedding_dim={self.embedding_dim}, "
                f"size={stats['current_size']}, "
                f"capacity={stats['current_capacity']}, "
                f"hit_rate={stats['hit_rate']:.2%})")