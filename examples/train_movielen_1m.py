import sys 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Tuple
import numpy as np
import hkv_embedding
from hkv_embedding.optimizer import HKVOptimizer, HKVAdamOptimizer, HKVAdagrad

file_path = sys.argv[1]

class MovieLensDataset(Dataset):
    """
    MovieLens 数据集类，用于处理 ratings.dat 格式的数据
    数据格式: UserID::MovieID::Rating::Timestamp
    """
    
    def __init__(self, ratings_file: str, sep: str = "::"):
        """
        初始化数据集
        
        Args:
            ratings_file: ratings.dat 文件路径
            sep: 分隔符，默认为 "::"
        """
        self.data = pd.read_csv(
            ratings_file,
            sep=sep,
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python'  # 使用python引擎处理自定义分隔符
        )
        
        # 转换为类别ID（从0开始）
        self.user_ids = self.data['user_id'].values
        self.movie_ids = self.data['movie_id'].values
        self.ratings = self.data['rating'].values
        self.timestamps = self.data['timestamp'].values
        
        # 获取唯一用户和电影数量
        self.num_users = len(np.unique(self.user_ids))
        self.num_movies = len(np.unique(self.movie_ids))
        
        print(f"数据集加载完成:")
        print(f"  总记录数: {len(self.data)}")
        print(f"  用户数: {self.num_users}")
        print(f"  电影数: {self.num_movies}")
        print(f"  评分范围: {self.ratings.min()} - {self.ratings.max()}")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 索引
            
        Returns:
            (user_id, movie_id, rating, timestamp)
        """
        user_id = torch.tensor(self.user_ids[idx], dtype=torch.long)
        movie_id = torch.tensor(self.movie_ids[idx], dtype=torch.long)
        rating = torch.tensor(self.ratings[idx], dtype=torch.float32)
        timestamp = torch.tensor(self.timestamps[idx], dtype=torch.long)
        
        return user_id, movie_id, rating, timestamp

def load_movielens_data(ratings_file: str, batch_size: int = 4096, 
                       shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    加载 MovieLens 数据并创建 DataLoader
    
    Args:
        ratings_file: ratings.dat 文件路径
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载工作线程数
        
    Returns:
        DataLoader 对象
    """
    dataset = MovieLensDataset(ratings_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader

class MovieLensDeepFMDataset(Dataset):
    """
    适用于 DeepFM 模型的 MovieLens 数据集
    将用户ID和电影ID作为稀疏特征处理
    """
    
    def __init__(self, ratings_file: str, sep: str = "::"):
        """
        初始化数据集
        
        Args:
            ratings_file: ratings.dat 文件路径
            sep: 分隔符，默认为 "::"
        """
        self.data = pd.read_csv(
            ratings_file,
            sep=sep,
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        # 为了适配 DeepFM 模型，我们需要将评分转换为二分类标签
        # 这里假设评分>=4为正样本(1)，评分<4为负样本(0)
        self.data['label'] = (self.data['rating'] >= 4).astype(int)
        
        self.user_ids = self.data['user_id'].values
        self.movie_ids = self.data['movie_id'].values
        self.labels = self.data['label'].values
        
        # 获取特征维度
        self.num_users = len(np.unique(self.user_ids))
        self.num_movies = len(np.unique(self.movie_ids))
        
        print(f"DeepFM数据集加载完成:")
        print(f"  总记录数: {len(self.data)}")
        print(f"  用户数: {self.num_users}")
        print(f"  电影数: {self.num_movies}")
        print(f"  正样本比例: {self.labels.mean():.2%}")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 索引
            
        Returns:
            (user_id, movie_id, label) 用于 DeepFM 模型
        """
        # 注意：这里返回的是原始ID，模型内部会处理映射
        user_id = torch.tensor(self.user_ids[idx], dtype=torch.long)
        movie_id = torch.tensor(self.movie_ids[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return user_id, movie_id, label

def load_movielens_for_deepfm(ratings_file: str, batch_size: int = 4096, 
                             shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    加载 MovieLens 数据用于 DeepFM 模型训练
    
    Args:
        ratings_file: ratings.dat 文件路径
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载工作线程数
        
    Returns:
        DataLoader 对象
    """
    dataset = MovieLensDeepFMDataset(ratings_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader



class DeepFMModel(nn.Module):
    """
    DeepFM-style model using HKV Embedding for sparse features.
    
    Suitable for CTR prediction with:
    - User ID (billions of possible values)
    - Item ID (millions of possible values)
    - Categorical features (variable cardinality)
    """
    
    def __init__(self, 
                 num_sparse_fields: int,
                 embedding_dim: int = 64,
                 mlp_dims: list = [256, 128, 64],
                 max_capacity_per_field: int = 100000000,  # 100M per field
                 max_hbm_gb_per_field: int = 4):
        super().__init__()
        
        self.num_sparse_fields = num_sparse_fields
        self.embedding_dim = embedding_dim
        
        # Use MultiTableHKVEmbedding for multiple feature fields
        self.sparse_embeddings = hkv_embedding.MultiTableHKVEmbedding(
            num_tables=num_sparse_fields,
            embedding_dim=embedding_dim,
            max_capacity_per_table=max_capacity_per_field,
            init_capacity_per_table=max_capacity_per_field // 100,
            max_hbm_gb_per_table=max_hbm_gb_per_field,
            device='cuda',
            shared_optimizer=True,
            debug_print=True
        )
        
        # FM interaction layer (no learnable parameters, just computation)
        
        # Deep MLP layers
        mlp_input_dim = num_sparse_fields * embedding_dim
        layers = []
        prev_dim = mlp_input_dim
        for dim in mlp_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, sparse_indices_list: list):
        """
        Forward pass.
        
        Args:
            sparse_indices_list: List of index tensors, one per sparse field
            
        Returns:
            Prediction logits
        """
        # Get embeddings for all sparse fields
        embeddings_list = self.sparse_embeddings(sparse_indices_list)
        # print(embeddings_list[:10])
        # Stack embeddings: [batch, num_fields, dim]
        stacked = torch.stack(embeddings_list, dim=1)
        batch_size = stacked.size(0)
        
        # FM component: sum of pairwise interactions
        # (sum(x))^2 - sum(x^2) / 2
        sum_square = torch.sum(stacked, dim=1) ** 2
        square_sum = torch.sum(stacked ** 2, dim=1)
        fm_out = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)
        
        # Deep component
        mlp_input = stacked.view(batch_size, -1)
        deep_out = self.mlp(mlp_input)
        
        # Combine FM and Deep
        logits = fm_out + deep_out
        
        return logits.squeeze(-1)


# 修改 DeepFM 模型以适应 MovieLens 数据
class MovieLensDeepFMModel(DeepFMModel):
    """
    适配 MovieLens 数据的 DeepFM 模型
    """
    
    def __init__(self, 
                 num_users: int,
                 num_movies: int,
                 embedding_dim: int = 64,
                 mlp_dims: list = [256, 128, 64],
                 max_capacity_per_field: int = 100000000):
        """
        初始化 MovieLens DeepFM 模型
        
        Args:
            num_users: 用户数量
            num_movies: 电影数量
            embedding_dim: 嵌入维度
            mlp_dims: MLP 层维度
            max_capacity_per_field: 每个字段的最大容量
        """
        # 两个稀疏字段：用户ID和电影ID
        super().__init__(
            num_sparse_fields=2,
            embedding_dim=embedding_dim,
            mlp_dims=mlp_dims,
            max_capacity_per_field=max_capacity_per_field,
            max_hbm_gb_per_field=4
        )
        
        self.num_users = num_users
        self.num_movies = num_movies
    
    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            user_ids: 用户ID张量
            movie_ids: 电影ID张量
            
        Returns:
            预测 logits
        """
        # 构建稀疏特征列表
        sparse_indices_list = [user_ids, movie_ids]
        
        # 调用父类的 forward 方法
        return super().forward(sparse_indices_list)

def train_deepfm_with_movielens(ratings_file: str, num_epochs: int = 5):
    """
    使用 MovieLens 数据训练 DeepFM 模型
    
    Args:
        ratings_file: ratings.dat 文件路径
        num_epochs: 训练轮数
    """
    print("=" * 60)
    print("使用 MovieLens 数据训练 DeepFM 模型")
    print("=" * 60)
    
    # 加载数据集以获取统计数据
    temp_dataset = MovieLensDeepFMDataset(ratings_file)
    num_users = temp_dataset.num_users
    num_movies = temp_dataset.num_movies
    
    # 创建模型
    model = MovieLensDeepFMModel(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=32,
        mlp_dims=[128, 64, 32],
        max_capacity_per_field=max(num_users, num_movies) + 1000
    )
    model = model.cuda()
    
    # 创建数据加载器
    dataloader = load_movielens_for_deepfm(
        ratings_file, 
        batch_size=4096, 
        shuffle=True
    )
    
    # 创建优化器
    pytorch_params = list(model.mlp.parameters())
    pytorch_optimizer = torch.optim.Adam(pytorch_params, lr=0.001)
    
    hkv_optimizer = HKVAdamOptimizer(
        model.sparse_embeddings.get_all_tables(),
        lr=0.001,
        betas=(0.9, 0.999),
        state_hbm_gb_per_embedding=1
    )
    
    # 损失函数
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (user_ids, movie_ids, labels) in enumerate(dataloader):
            # 移动数据到GPU
            user_ids = user_ids.cuda()
            movie_ids = movie_ids.cuda()
            labels = labels.cuda()
            
            # 前向传播
            logits = model(user_ids, movie_ids)
            loss = criterion(logits, labels)
            
            # 反向传播
            pytorch_optimizer.zero_grad()
            hkv_optimizer.zero_grad()
            
            loss.backward()
            
            # 更新参数
            pytorch_optimizer.step()
            hkv_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        # 打印嵌入统计信息
        for i, table in enumerate(model.sparse_embeddings.get_all_tables()):
            field_name = "User" if i == 0 else "Movie"
            stats = table.get_statistics()
            print(f"  {field_name} Field: {stats['current_size']} entries, "
                  f"hit_rate: {stats['hit_rate']:.2%}")

# 使用示例
if __name__ == "__main__":
    # 示例用法
    ratings_file = file_path  # 替换为实际文件路径
    
    # 方式1: 直接使用数据集
    # dataloader = load_movielens_for_deepfm(ratings_file)
    # for batch_idx, (user_ids, movie_ids, labels) in enumerate(dataloader):
    #     print(f"Batch {batch_idx}:")
    #     print(f"  Users: {user_ids[:5]}")  # 显示前5个
    #     print(f"  Movies: {movie_ids[:5]}")
    #     print(f"  Labels: {labels[:5]}")
    #     break
    
    # 方式2: 训练模型
    train_deepfm_with_movielens(ratings_file, num_epochs=1)
    # pass