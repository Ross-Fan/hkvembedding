import torch
import torch.nn as nn
import torch.nn.functional as F
import hkv_embedding
from hkv_embedding.optimizer import HKVOptimizer, HKVAdamOptimizer

class RecommendationModel(nn.Module):
    """
    使用HKV Embedding的推荐模型示例
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        
        # 使用HKV Embedding
        self.user_embedding = hkv_embedding.HierarchicalHashEmbedding(
            embedding_dim=embedding_dim,
            max_capacity=num_users * 2,  # 允许动态扩展
            init_capacity=min(num_users, 100000),
            max_hbm_gb=4,
            learning_rate=0.01
        )
        
        self.item_embedding = hkv_embedding.HierarchicalHashEmbedding(
            embedding_dim=embedding_dim,
            max_capacity=num_items * 2,
            init_capacity=min(num_items, 100000),
            max_hbm_gb=4,
            learning_rate=0.01
        )
        
        # 预测层
        self.fc = nn.Linear(embedding_dim * 2, 1)
    
    def forward(self, user_ids, item_ids):
        # 获取嵌入向量（支持自动微分）
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 拼接特征
        features = torch.cat([user_emb, item_emb], dim=-1)
        
        # 预测评分
        scores = self.fc(features)
        return scores.squeeze()

def train_model():
    """训练示例"""
    # 模型参数
    num_users = 1000000
    num_items = 500000
    embedding_dim = 64
    batch_size = 1024
    num_epochs = 10
    
    # 创建模型
    model = RecommendationModel(num_users, num_items, embedding_dim)
    
    # 创建混合优化器
    # 标准PyTorch参数使用Adam
    pytorch_params = [p for p in model.parameters() if p.requires_grad]
    pytorch_optimizer = torch.optim.Adam(pytorch_params, lr=0.001)
    
    # HKV Embedding使用自定义优化器
    hkv_optimizer = HKVAdamOptimizer(
        [model.user_embedding, model.item_embedding], 
        lr=0.001
    )
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # 模拟训练数据
        for batch_idx in range(100):  # 100个批次
            # 生成随机数据
            user_ids = torch.randint(0, num_users, (batch_size,))
            item_ids = torch.randint(0, num_items, (batch_size,))
            ratings = torch.randn(batch_size) * 2 + 3  # 模拟评分
            
            # 前向传播
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            
            # 反向传播
            pytorch_optimizer.zero_grad()
            hkv_optimizer.zero_grad()
            
            loss.backward()
            
            # 参数更新
            pytorch_optimizer.step()
            hkv_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                # 打印embedding统计信息
                user_stats = model.user_embedding.get_statistics()
                item_stats = model.item_embedding.get_statistics()
                print(f"  User Embedding: {user_stats['current_size']} entries, "
                      f"hit_rate: {user_stats['hit_rate']:.2%}")
                print(f"  Item Embedding: {item_stats['current_size']} entries, "
                      f"hit_rate: {item_stats['hit_rate']:.2%}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")

def test_gradient_flow():
    """测试梯度流动"""
    print("=== 测试HKV Embedding梯度流动 ===")
    
    # 创建简单模型
    embedding = hkv_embedding.HierarchicalHashEmbedding(
        embedding_dim=8,
        max_capacity=1000,
        init_capacity=100,
        max_hbm_gb=1,
        learning_rate=0.1
    )
    
    # 创建优化器
    optimizer = HKVOptimizer([embedding], lr=0.1)
    
    # 测试数据
    indices = torch.tensor([1, 2, 3, 4, 5])
    target = torch.randn(5, 8)
    
    print("初始统计:", embedding.get_statistics())
    
    # 训练几步
    for step in range(5):
        optimizer.zero_grad()
        
        # 前向传播
        embeddings = embedding(indices)
        loss = F.mse_loss(embeddings, target)
        
        print(f"Step {step}: Loss = {loss.item():.4f}")
        
        # 反向传播
        loss.backward()
        
        # 检查是否有梯度
        stats = embedding.get_statistics()
        print(f"  Pending gradients: {stats['pending_gradients']}")
        
        # 更新参数
        optimizer.step()
        
        print(f"  After update: {embedding.get_statistics()['pending_gradients']} pending gradients")
    
    print("最终统计:", embedding.get_statistics())

if __name__ == "__main__":
    # 测试梯度流动
    test_gradient_flow()
    
    # 训练完整模型
    # train_model()
