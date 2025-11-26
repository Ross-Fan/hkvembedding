import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from . import hkv_core

class HKVEmbeddingFunction(torch.autograd.Function):
    """
    HKV Embedding的自定义autograd函数
    支持前向传播和反向传播
    """
    
    @staticmethod
    def forward(ctx, indices, embedding_layer):
        """
        前向传播
        
        Args:
            ctx: autograd上下文
            indices: 输入的索引张量
            embedding_layer: HierarchicalHashEmbedding实例
        
        Returns:
            embeddings: 嵌入向量
        """
        # 保存上下文信息用于反向传播
        ctx.embedding_layer = embedding_layer
        ctx.save_for_backward(indices)
        
        # 执行前向传播
        with torch.no_grad():
            embeddings = embedding_layer._forward_impl(indices)
        
        return embeddings
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        
        Args:
            ctx: autograd上下文
            grad_output: 输出的梯度
        
        Returns:
            tuple: (indices的梯度, embedding_layer的梯度)
        """
        indices, = ctx.saved_tensors
        embedding_layer = ctx.embedding_layer
        
        # 累积梯度到embedding层
        embedding_layer._accumulate_gradients(indices, grad_output)
        
        # indices不需要梯度，embedding_layer也不需要梯度（参数更新通过内部机制）
        return None, None

class HierarchicalHashEmbedding(nn.Module):
    """
    基于HierarchicalKV的PyTorch嵌入层，支持自动微分
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 max_capacity: int = 1000000,
                 init_capacity: int = 100000,
                 max_hbm_gb: int = 16,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32,
                 init_std: float = None,
                 learning_rate: float = 0.01,
                 weight_decay: float = 0.0):
        """
        初始化HierarchicalHashEmbedding
        
        Args:
            embedding_dim: 嵌入向量维度
            max_capacity: 最大容量
            init_capacity: 初始容量
            max_hbm_gb: 最大GPU内存使用量(GB)
            device: 设备类型
            dtype: 数据类型
            init_std: 初始化标准差
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_capacity = max_capacity
        self.init_capacity = init_capacity
        self.device = device
        self.dtype = dtype
        self.init_std = init_std or np.sqrt(2.0 / embedding_dim)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 创建HKV哈希表
        try:
            self.hashtable = hkv_core.HashTable(
                init_capacity, max_capacity, embedding_dim, max_hbm_gb
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create HashTable: {e}")
        
        # 梯度累积器
        self.gradient_accumulator = {}  # key -> gradient tensor
        self.gradient_count = {}        # key -> count for averaging
        
        # 统计信息
        self.hit_count = 0
        self.miss_count = 0
        self.total_queries = 0
        
        # 注册为模块参数（用于优化器识别）
        self.register_buffer('_dummy_param', torch.zeros(1, requires_grad=False))
    
    def forward(self, indices):
        """
        前向传播（支持自动微分）
        
        Args:
            indices: 输入索引张量
            
        Returns:
            embeddings: 嵌入向量张量
        """
        if indices.numel() == 0:
            return torch.empty(0, self.embedding_dim, dtype=self.dtype, device=self.device)
        
        original_shape = indices.shape
        indices_flat = indices.flatten()
        
        # 使用自定义autograd函数
        embeddings = HKVEmbeddingFunction.apply(indices_flat, self)
        
        # 恢复原始形状
        if len(original_shape) == 1:
            return embeddings
        else:
            return embeddings.view(*original_shape, self.embedding_dim)
    
    def _forward_impl(self, indices):
        """
        实际的前向传播实现
        
        Args:
            indices: 扁平化的索引张量
            
        Returns:
            embeddings: 嵌入向量张量
        """
        # 转换为numpy数组
        indices_np = indices.cpu().numpy().astype(np.uint64)
        
        try:
            # 调用HKV查找或插入
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
            self._update_embeddings_impl(indices[new_mask], new_embeddings)
        
        # 更新统计
        self.hit_count += found_flags.sum()
        self.miss_count += len(found_flags) - found_flags.sum()
        self.total_queries += len(found_flags)
        
        return embeddings
    
    def _accumulate_gradients(self, indices, grad_output):
        """
        累积梯度
        
        Args:
            indices: 索引张量
            grad_output: 输出梯度
        """
        if grad_output is None or indices.numel() == 0:
            return
        
        # 确保梯度在CPU上进行累积（避免GPU内存碎片）
        indices_cpu = indices.cpu()
        grad_cpu = grad_output.detach().cpu()
        
        # 按索引累积梯度
        for i, idx in enumerate(indices_cpu):
            idx_item = idx.item()
            
            if idx_item in self.gradient_accumulator:
                self.gradient_accumulator[idx_item] += grad_cpu[i]
                self.gradient_count[idx_item] += 1
            else:
                self.gradient_accumulator[idx_item] = grad_cpu[i].clone()
                self.gradient_count[idx_item] = 1
    
    def step(self):
        """
        执行一步梯度更新（类似optimizer.step()）
        """
        if not self.gradient_accumulator:
            return
        
        # 准备批量更新
        keys = []
        gradients = []
        
        for key, grad in self.gradient_accumulator.items():
            # 平均梯度
            avg_grad = grad / self.gradient_count[key]
            
            # 应用权重衰减
            if self.weight_decay > 0:
                # 获取当前嵌入
                current_embedding = self._get_embedding(key)
                if current_embedding is not None:
                    avg_grad += self.weight_decay * current_embedding
            
            keys.append(key)
            gradients.append(avg_grad)
        
        if keys:
            # 批量更新嵌入
            self._update_embeddings_with_gradients(keys, gradients)
        
        # 清空梯度累积器
        self.gradient_accumulator.clear()
        self.gradient_count.clear()
    
    def _get_embedding(self, key):
        """获取单个key的嵌入向量"""
        try:
            keys_np = np.array([key], dtype=np.uint64)
            embeddings_np, found = self.hashtable.find(keys_np)
            if found[0]:
                return torch.from_numpy(embeddings_np[0])
            return None
        except:
            return None
    
    def _update_embeddings_with_gradients(self, keys, gradients):
        """
        使用梯度更新嵌入向量
        
        Args:
            keys: 键列表
            gradients: 梯度列表
        """
        try:
            # 获取当前嵌入
            keys_np = np.array(keys, dtype=np.uint64)
            current_embeddings_np, found = self.hashtable.find(keys_np)
            
            # 转换为torch张量
            current_embeddings = torch.from_numpy(current_embeddings_np)
            
            # 应用梯度更新
            updated_embeddings = current_embeddings.clone()
            for i, (key, grad) in enumerate(zip(keys, gradients)):
                if found[i]:  # 只更新存在的键
                    updated_embeddings[i] -= self.learning_rate * grad
            
            # 更新回哈希表
            updated_embeddings_np = updated_embeddings.numpy().astype(np.float32)
            self.hashtable.insert_or_assign(keys_np, updated_embeddings_np)
            
        except Exception as e:
            print(f"Warning: Failed to update embeddings with gradients: {e}")
    
    def _update_embeddings_impl(self, indices, embeddings):
        """更新哈希表中的嵌入向量（内部实现）"""
        if indices.numel() == 0:
            return
            
        indices_np = indices.cpu().numpy().astype(np.uint64)
        embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)
        
        try:
            self.hashtable.insert_or_assign(indices_np, embeddings_np)
        except Exception as e:
            raise RuntimeError(f"HKV insert_or_assign failed: {e}")
    
    def zero_grad(self):
        """清空梯度（类似optimizer.zero_grad()）"""
        self.gradient_accumulator.clear()
        self.gradient_count.clear()
    
    def parameters(self):
        """返回可训练参数（为了兼容PyTorch优化器）"""
        # 返回一个虚拟参数，实际参数管理由内部处理
        return [self._dummy_param]
    
    def named_parameters(self):
        """返回命名参数"""
        yield 'hkv_embeddings', self._dummy_param
    
    def state_dict(self):
        """保存状态字典"""
        # 这里可以实现保存HKV哈希表的逻辑
        return {
            'embedding_dim': self.embedding_dim,
            'max_capacity': self.max_capacity,
            'init_capacity': self.init_capacity,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'total_queries': self.total_queries,
        }
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.embedding_dim = state_dict.get('embedding_dim', self.embedding_dim)
        self.max_capacity = state_dict.get('max_capacity', self.max_capacity)
        self.init_capacity = state_dict.get('init_capacity', self.init_capacity)
        self.learning_rate = state_dict.get('learning_rate', self.learning_rate)
        self.weight_decay = state_dict.get('weight_decay', self.weight_decay)
        self.hit_count = state_dict.get('hit_count', 0)
        self.miss_count = state_dict.get('miss_count', 0)
        self.total_queries = state_dict.get('total_queries', 0)
    
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
            'embedding_dim': self.embedding_dim,
            'pending_gradients': len(self.gradient_accumulator),
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
        }
    
    def __repr__(self):
        stats = self.get_statistics()
        return (f"HierarchicalHashEmbedding("
                f"embedding_dim={self.embedding_dim}, "
                f"size={stats['current_size']}, "
                f"capacity={stats['current_capacity']}, "
                f"hit_rate={stats['hit_rate']:.2%}, "
                f"lr={self.learning_rate}, "
                f"pending_grads={stats['pending_gradients']})")
