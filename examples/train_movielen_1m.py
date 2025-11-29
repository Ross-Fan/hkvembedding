import sys 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Tuple
import numpy as np
import kv_embedding
# import hkv_embedding
# from hkv_embedding.optimizer import HKVOptimizer, HKVAdamOptimizer, HKVAdagrad

file_path = sys.argv[1]

class MovieLensDataset(Dataset):
    """
    Dataset class for MovieLens 1M data
    Data format: UserID::MovieID::Rating::Timestamp
    """
    def __init__(self, file_path: str):
        # Read the data with '::' as separator
        self.data = pd.read_csv(
            file_path, 
            sep='::', 
            header=None, 
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_id = torch.tensor(row['user_id'], dtype=torch.long)
        item_id = torch.tensor(row['item_id'], dtype=torch.long)
        # rating = torch.tensor(row['rating'], dtype=torch.float32)
        rating = torch.tensor(int(row['rating']) - 1, dtype=torch.long)
        return user_id, item_id, rating

class MatrixFactorization(nn.Module):
    """
    Matrix Factorization model for rating prediction
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, num_classes: int = 5):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # self.user_embedding = kv_embedding.KVEmbedding(embedding_dim, 0.0, 0.001)
        # self.item_embedding = kv_embedding.KVEmbedding(embedding_dim, 0.0, 0.001)

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
        self.global_bias = nn.Parameter(torch.zeros(1))

        # 分类头 - 将交互特征映射到类别分数
        self.classifier = nn.Linear(embedding_dim + 2, num_classes) 
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        # print("user_ids:\n", user_ids)
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Get biases
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        
        # Compute dot product and add biases
        # rating = (user_emb * item_emb).sum(dim=1) + user_b + item_b + self.global_bias
        # 组合所有特征
        # 计算交互特征 (element-wise product)
        interaction = user_emb * item_emb  # [batch_size, embedding_dim]
        # 将交互特征和偏置拼接
        combined_features = torch.cat([
            interaction, 
            user_b.unsqueeze(1), 
            item_b.unsqueeze(1)
        ], dim=1)  # [batch_size, embedding_dim + 2]
        
        # 通过分类器得到各类别分数
        logits = self.classifier(combined_features)  # [batch_size, num_classes]
        
        return logits

def train_model(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, epochs: int = 10):
    """
    Training function for the model
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for batch_idx, (user_ids, item_ids, ratings) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(user_ids, item_ids)
            # print("logits:", logits)
            loss = criterion(logits, ratings)
            
            # Backward pass
            loss.backward()
            # Print gradients of user and item embeddings
            # if batch_idx % 100 == 0:
            #     user_emb_grad = model.user_embedding.weight.grad
            #     item_emb_grad = model.item_embedding.weight.grad
            #     print(f'User embedding gradient - Shape: {user_emb_grad.shape}, Norm: {user_emb_grad.norm().item():.6f}')
            #     print(f'Item embedding gradient - Shape: {item_emb_grad.shape}, Norm: {item_emb_grad.norm().item():.6f}')
            #     # Uncomment below to see actual gradient values (might be verbose)
            #     print(f'User embedding gradients sample: {user_emb_grad[:5]}')  # First 5 rows
            #     print(f'Item embedding gradients sample: {item_emb_grad[:5]}')  # First 5 rows
            
            optimizer.step()
            
            total_loss += loss.item()

            # 计算准确率
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == ratings).sum().item()
            total_predictions += ratings.size(0)
            
            if batch_idx % 100 == 0:
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
                
                
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')

def main():
    file_path = sys.argv[1]
    
    # Create dataset and dataloader
    dataset = MovieLensDataset(file_path)
    
    # Get number of unique users and items
    num_users = dataset.data['user_id'].max() + 1
    num_items = dataset.data['item_id'].max() + 1
    
    print(f"Dataset loaded with {len(dataset)} samples")
    print(f"Number of users: {num_users}, Number of items: {num_items}")
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Create model
    model = MatrixFactorization(num_users, num_items, embedding_dim=8)
    
    # Define loss function and optimizer
    criterion =  nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, dataloader, optimizer, criterion, epochs=1)

if __name__ == "__main__":
    main()