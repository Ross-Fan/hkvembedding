"""
HKV Embedding Training Example

Demonstrates how to use HKV Embedding with PyTorch for large-scale
recommendation model training with billions of unique IDs.
"""
import os, time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import hkv_embedding
from torch.utils.data import Dataset, DataLoader
from hkv_embedding.optimizer import HKVOptimizer, HKVAdamOptimizer, HKVAdagrad
from moviedata import MovieLensDataset
from tfrecord_dataloader import StreamingTFRecordDataset

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
                 num_classes: int = 5):
        super().__init__()
        
        self.num_sparse_fields = num_sparse_fields
        self.embedding_dim = embedding_dim
        
        # Use MultiTableHKVEmbedding for multiple feature fields
        self.sparse_embeddings = hkv_embedding.HierarchicalHashEmbedding(
            embedding_dim = embedding_dim,
            max_capacity = 100000000,
            init_capacity = 1000000,
            max_hbm_gb = 4,
            device='cuda'
        )
        # self.sparse_embeddings = nn.Embedding(num_embeddings=1000000, embedding_dim=embedding_dim)
        
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
        # layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

        # 分类头 - 将交互特征映射到类别分数
        self.classifier = nn.Linear(mlp_dims[-1], num_classes) 
    
    def forward(self, discrete_features: torch.Tensor, sequence_features: torch.Tensor):
        """
        Forward pass.
        
        Args:
            sparse_indices_list: List of index tensors, one per sparse field
            
        Returns:
            Prediction logits
        """
        # Get embeddings for all sparse fields
        dis_emb = self.sparse_embeddings(discrete_features)
        seq_emb = self.sparse_embeddings(sequence_features)
        print("dis_emb", dis_emb.shape)
        print("seq_emb", seq_emb.shape)
        # embeddings_list = self.sparse_embeddings(sparse_indices_list)
        # print(embeddings_list[:10])
        # Stack embeddings: [batch, num_fields, dim]
        # concat_emb = torch.concat([user_emb, item_emb], dim=-1)
        
        
        # FM component: sum of pairwise interactions
        # (sum(x))^2 - sum(x^2) / 2
        # sum_square = torch.sum(stacked, dim=1) ** 2
        # square_sum = torch.sum(stacked ** 2, dim=1)
        # fm_out = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)
        
        # Deep component
        # mlp_input = concat_emb
        # deep_out = self.mlp(mlp_input)
        
        # Combine FM and Deep
        logits = 0
        
        return logits


class TwoTowerModel(nn.Module):
    """
    Two-Tower model for retrieval tasks.
    
    User tower and item tower with separate HKV embeddings.
    """
    
    def __init__(self,
                 user_embedding_dim: int = 64,
                 item_embedding_dim: int = 64,
                 tower_dims: list = [128, 64],
                 max_users: int = 1000000000,  # 1B users
                 max_items: int = 100000000,   # 100M items
                 user_hbm_gb: int = 16,
                 item_hbm_gb: int = 8):
        super().__init__()
        
        # User embedding with HKV
        self.user_embedding = hkv_embedding.HierarchicalHashEmbedding(
            embedding_dim=user_embedding_dim,
            max_capacity=max_users,
            init_capacity=max_users // 1000,
            max_hbm_gb=user_hbm_gb,
            device='cuda'
        )
        
        # Item embedding with HKV
        self.item_embedding = hkv_embedding.HierarchicalHashEmbedding(
            embedding_dim=item_embedding_dim,
            max_capacity=max_items,
            init_capacity=max_items // 1000,
            max_hbm_gb=item_hbm_gb,
            device='cuda'
        )
        
        # User tower
        user_layers = []
        prev_dim = user_embedding_dim
        for dim in tower_dims:
            user_layers.append(nn.Linear(prev_dim, dim))
            user_layers.append(nn.ReLU())
            prev_dim = dim
        self.user_tower = nn.Sequential(*user_layers)
        
        # Item tower
        item_layers = []
        prev_dim = item_embedding_dim
        for dim in tower_dims:
            item_layers.append(nn.Linear(prev_dim, dim))
            item_layers.append(nn.ReLU())
            prev_dim = dim
        self.item_tower = nn.Sequential(*item_layers)
    
    def forward(self, user_ids, item_ids):
        """
        Forward pass.
        
        Args:
            user_ids: User ID tensor
            item_ids: Item ID tensor
            
        Returns:
            Similarity scores
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Pass through towers
        user_vec = self.user_tower(user_emb)
        item_vec = self.item_tower(item_emb)
        
        # Normalize for cosine similarity
        user_vec = F.normalize(user_vec, p=2, dim=-1)
        item_vec = F.normalize(item_vec, p=2, dim=-1)
        
        # Dot product similarity
        scores = torch.sum(user_vec * item_vec, dim=-1)
        
        return scores
    
    def get_hkv_embeddings(self):
        """Return list of HKV embedding layers for optimizer."""
        return [self.user_embedding, self.item_embedding]


def train_deepfm(dataloader: DataLoader):
    """Train DeepFM model example."""
    print("=" * 60)
    print("Training DeepFM with HKV Embedding")
    print("=" * 60)
    
    # Model config
    num_sparse_fields = 2
    embedding_dim = 32
    batch_size = 16
    num_epochs = 1
    
    # Create model
    model = DeepFMModel(
        num_sparse_fields=num_sparse_fields,
        embedding_dim=embedding_dim,
        mlp_dims=[128, 64, 32],
        
    )
    model = model.cuda()
    print("mode gets_all_tables", model.sparse_embeddings)
    # Create optimizers
    # PyTorch optimizer for MLP layers
    pytorch_params = list(model.mlp.parameters())
    pytorch_optimizer = torch.optim.Adam(pytorch_params, lr=0.001)
    
    # HKV Adam optimizer for embeddings (GPU-backed states)
    hkv_optimizer = HKVAdamOptimizer(
        model.sparse_embeddings,
        lr=0.001,
        betas=(0.9, 0.999),
        state_hbm_gb_per_embedding=1  # 1GB for Adam states per embedding
    )
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        correct_predictions = 0  # 初始化正确预测计数
        total_samples = 0        # 初始化总样本数
        
        for batch_idx, (discrete_features, seq_features, labels) in enumerate(dataloader):  # 100 batches per epoch
            batch_start_time = time.time()
            # 确保所有张量都在同一个设备上 (CUDA)
            # discrete_features = discrete_features.cuda()
            # seq_features = seq_features.cuda()
            # ratings = ratings.cuda()  # 关键修改：将 ratings 移动到 CUDA 设备
            
            # Forward
            forward_start = time.time()
            logits = model(discrete_features, seq_features)

            play_click_1m = labels['play_1m_label'].to('cuda')
            forward_time = time.time() - forward_start
            
            
            # Backward
            back_start_time = time.time()
            loss = criterion(logits, play_click_1m)
            pytorch_optimizer.zero_grad()
            hkv_optimizer.zero_grad()
            
            loss.backward()
            
            # 反向后、更新前：查看 HKV buffer 状态
            # for i, table in enumerate(model.sparse_embeddings.get_all_tables()):
            #     print(f' Batch {batch_idx} Field {i} pending_grads (post-backward):', table.get_pending_gradient_count())
            # Update
            pytorch_optimizer.step()
            hkv_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            backward_time = time.time() - back_start_time
            
            batch_time = time.time() - batch_start_time
            if batch_idx % 25 == 0:
                batch_accuracy = (predicted == ratings).sum().item() / ratings.size(0)
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f},  Accuracy: {batch_accuracy:.4f},"
                      f"Forward: {forward_time*1000:.2f}ms, "
                      f"Backward: {(backward_time)*1000:.2f}ms, "
                      f"Total: {batch_time*1000:.2f}ms")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        # Print embedding statistics
        # for i, table in enumerate(model.sparse_embeddings.get_all_tables()):
        #     stats = table.get_statistics()
        #     print(f"  Field {i}: {stats['current_size']} entries, "
        #           f"hit_rate: {stats['hit_rate']:.2%}, "
        #           f"pending_grads: {stats['pending_gradients']}")


def train_two_tower():
    """Train Two-Tower model example."""
    print("=" * 60)
    print("Training Two-Tower with HKV Embedding")
    print("=" * 60)
    
    # Model config
    batch_size = 2048
    num_epochs = 5
    
    # Create model (smaller scale for example)
    model = TwoTowerModel(
        user_embedding_dim=64,
        item_embedding_dim=64,
        tower_dims=[128, 64],
        max_users=10000000,   # 10M users
        max_items=1000000,    # 1M items
        user_hbm_gb=4,
        item_hbm_gb=2
    )
    model = model.cuda()
    
    # Optimizers
    tower_params = list(model.user_tower.parameters()) + list(model.item_tower.parameters())
    pytorch_optimizer = torch.optim.Adam(tower_params, lr=0.001)
    
    # Use Adagrad for embeddings (good for sparse features)
    hkv_optimizer = HKVAdagrad(
        model.get_hkv_embeddings(),
        lr=0.1,
        state_hbm_gb_per_embedding=1
    )
    
    # Contrastive loss (in-batch negatives)
    temperature = 0.1
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx in range(100):
            # Generate random data
            user_ids = torch.randint(0, 10000000, (batch_size,), device='cuda')
            pos_item_ids = torch.randint(0, 1000000, (batch_size,), device='cuda')
            
            # Forward
            scores = model(user_ids, pos_item_ids)
            
            # In-batch negative sampling: compare each user with all items in batch
            user_emb = model.user_embedding(user_ids)
            item_emb = model.item_embedding(pos_item_ids)
            
            user_vec = model.user_tower(user_emb)
            item_vec = model.item_tower(item_emb)
            
            user_vec = F.normalize(user_vec, p=2, dim=-1)
            item_vec = F.normalize(item_vec, p=2, dim=-1)
            
            # Compute similarity matrix
            similarity = torch.matmul(user_vec, item_vec.T) / temperature
            
            # Labels: diagonal elements are positive pairs
            labels = torch.arange(batch_size, device='cuda')
            
            # Cross-entropy loss
            loss = F.cross_entropy(similarity, labels)
            
            # Backward
            pytorch_optimizer.zero_grad()
            hkv_optimizer.zero_grad()
            
            loss.backward()
            
            pytorch_optimizer.step()
            hkv_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 25 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        # Print statistics
        user_stats = model.user_embedding.get_statistics()
        item_stats = model.item_embedding.get_statistics()
        print(f"  User Embedding: {user_stats['current_size']} entries, "
              f"hit_rate: {user_stats['hit_rate']:.2%}")
        print(f"  Item Embedding: {item_stats['current_size']} entries, "
              f"hit_rate: {item_stats['hit_rate']:.2%}")


def test_gradient_flow():
    """Test gradient flow through HKV Embedding."""
    print("=" * 60)
    print("Testing HKV Embedding Gradient Flow")
    print("=" * 60)
    
    # Create embedding
    embedding = hkv_embedding.HierarchicalHashEmbedding(
        embedding_dim=16,
        max_capacity=10000,
        init_capacity=1000,
        max_hbm_gb=1,
        learning_rate=0.1,
        grad_buffer_hbm_gb=1  # GPU-backed gradient buffer
    )
    
    # Create optimizer
    optimizer = HKVOptimizer([embedding], lr=0.1)
    
    # Test data
    indices = torch.tensor([1, 2, 3, 4, 5], device='cuda')
    target = torch.randn(5, 16, device='cuda')
    
    print(f"Initial stats: {embedding.get_statistics()}")
    
    # Training loop
    for step in range(10):
        optimizer.zero_grad()
        
        # Forward
        embeddings = embedding(indices)
        loss = F.mse_loss(embeddings, target)
        
        # Backward
        loss.backward()
        
        print(f"Step {step}: Loss = {loss.item():.4f}, "
              f"Pending gradients = {embedding.get_pending_gradient_count()}")
        
        # Update
        optimizer.step()
    
    print(f"Final stats: {embedding.get_statistics()}")


def test_billion_scale():
    """Test with billion-scale IDs (simulated)."""
    print("=" * 60)
    print("Testing Billion-Scale ID Support")
    print("=" * 60)
    
    # Create embedding with large capacity
    embedding = hkv_embedding.HierarchicalHashEmbedding(
        embedding_dim=32,
        max_capacity=1000000000,  # 1 billion capacity
        init_capacity=1000000,    # Start with 1M
        max_hbm_gb=8,
        grad_buffer_hbm_gb=2
    )
    
    optimizer = HKVAdamOptimizer(
        [embedding],
        lr=0.001,
        state_hbm_gb_per_embedding=4  # 4GB for Adam states
    )
    
    batch_size = 8192
    
    # Simulate training with random billion-scale IDs
    for step in range(20):
        optimizer.zero_grad()
        
        # Random IDs in billions range
        indices = torch.randint(
            0, 1000000000,  # 0 to 1 billion
            (batch_size,),
            device='cuda',
            dtype=torch.long
        )
        
        # Forward
        embeddings = embedding(indices)
        
        # Simple loss
        loss = embeddings.mean()
        loss.backward()
        
        optimizer.step()
        
        if step % 5 == 0:
            stats = embedding.get_statistics()
            print(f"Step {step}: Size = {stats['current_size']}, "
                  f"Load factor = {stats['load_factor']:.4f}")
    
    print(f"Final: {embedding.get_statistics()}")

def get_file_list(base_local_path: str):
    """获取 Parquet 文件列表"""
    files = []
    if os.path.exists(base_local_path):
        for d in os.listdir(base_local_path):
            if d.endswith(".tfrecord.gz"):
                files.append(os.path.join(base_local_path, d))
    files.sort()
    print(f"Found {len(files)} Parquet files.")
    return files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HKV Embedding Training Examples")
    parser.add_argument("--file_path", type=str, default="",
                        help="Path to the training file")
    
    args = parser.parse_args()

    # dataset = MovieLensDataset(args.file_path)
    # dataset = StreamingTFRecordDataset(args.file_path)
    # Get number of unique users and items
    # num_users = dataset.data['user_id'].max() + 1
    # num_items = dataset.data['item_id'].max() + 1
    
    # print(f"Dataset loaded with {len(dataset)} samples")
    # print(f"Number of users: {num_users}, Number of items: {num_items}")
    
    # Create data loader
    # dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    files = get_file_list(args.file_path)
    dataloader = StreamingTFRecordDataset(files, 0, 0, batch_size=1024)
    
    
    train_deepfm(dataloader)
    # parser.add_argument("--example", type=str, default="gradient",
    #                    choices=["gradient", "deepfm", "twotower", "billion"],
    #                    help="Which example to run")
    
    # args = parser.parse_args()
    
    # if args.example == "gradient":
    #     test_gradient_flow()
    # elif args.example == "deepfm":
    #     train_deepfm()
    # elif args.example == "twotower":
    #     train_two_tower()
    # elif args.example == "billion":
    #     test_billion_scale()
