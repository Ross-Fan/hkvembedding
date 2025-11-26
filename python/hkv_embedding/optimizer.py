import torch
from typing import List, Dict, Any

class HKVOptimizer:
    """
    HKV Embedding的自定义优化器
    """
    
    def __init__(self, hkv_embeddings: List, lr: float = 0.01, weight_decay: float = 0.0):
        """
        初始化优化器
        
        Args:
            hkv_embeddings: HierarchicalHashEmbedding实例列表
            lr: 学习率
            weight_decay: 权重衰减
        """
        self.hkv_embeddings = hkv_embeddings if isinstance(hkv_embeddings, list) else [hkv_embeddings]
        self.lr = lr
        self.weight_decay = weight_decay
        
        # 更新所有embedding的学习率
        for embedding in self.hkv_embeddings:
            embedding.learning_rate = lr
            embedding.weight_decay = weight_decay
    
    def step(self):
        """执行一步优化"""
        for embedding in self.hkv_embeddings:
            embedding.step()
    
    def zero_grad(self):
        """清空梯度"""
        for embedding in self.hkv_embeddings:
            embedding.zero_grad()
    
    def state_dict(self):
        """保存优化器状态"""
        return {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        }
    
    def load_state_dict(self, state_dict):
        """加载优化器状态"""
        self.lr = state_dict['lr']
        self.weight_decay = state_dict['weight_decay']
        
        # 更新所有embedding的参数
        for embedding in self.hkv_embeddings:
            embedding.learning_rate = self.lr
            embedding.weight_decay = self.weight_decay

class HKVAdamOptimizer:
    """
    HKV Embedding的Adam优化器
    """
    
    def __init__(self, hkv_embeddings: List, lr: float = 0.001, 
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8, 
                 weight_decay: float = 0.0):
        """
        初始化Adam优化器
        """
        self.hkv_embeddings = hkv_embeddings if isinstance(hkv_embeddings, list) else [hkv_embeddings]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        
        # Adam状态
        self.m_dict = {}  # 一阶矩估计
        self.v_dict = {}  # 二阶矩估计
    
    def step(self):
        """执行Adam优化步骤"""
        self.step_count += 1
        
        for embedding in self.hkv_embeddings:
            if not embedding.gradient_accumulator:
                continue
            
            # 准备更新
            keys = []
            gradients = []
            
            for key, grad in embedding.gradient_accumulator.items():
                # 平均梯度
                avg_grad = grad / embedding.gradient_count[key]
                
                # Adam更新
                if key not in self.m_dict:
                    self.m_dict[key] = torch.zeros_like(avg_grad)
                    self.v_dict[key] = torch.zeros_like(avg_grad)
                
                # 更新矩估计
                self.m_dict[key] = self.betas[0] * self.m_dict[key] + (1 - self.betas[0]) * avg_grad
                self.v_dict[key] = self.betas[1] * self.v_dict[key] + (1 - self.betas[1]) * (avg_grad ** 2)
                
                # 偏差校正
                m_hat = self.m_dict[key] / (1 - self.betas[0] ** self.step_count)
                v_hat = self.v_dict[key] / (1 - self.betas[1] ** self.step_count)
                
                # 计算更新
                update = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
                
                # 添加权重衰减
                if self.weight_decay > 0:
                    current_embedding = embedding._get_embedding(key)
                    if current_embedding is not None:
                        update += self.weight_decay * current_embedding
                
                keys.append(key)
                gradients.append(update)
            
            if keys:
                # 批量更新（这里直接使用更新值，不是梯度）
                embedding._apply_updates(keys, gradients)
            
            # 清空梯度
            embedding.zero_grad()
    
    def zero_grad(self):
        """清空梯度"""
        for embedding in self.hkv_embeddings:
            embedding.zero_grad()
