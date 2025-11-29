import sys
import torch 
import torch.nn as nn
from typing import List, Dict, Set
import numpy as np

class KVEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, mean: float, std: float, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.init_mean = mean
        self.init_std = std
        
        # Store registered keys separately
        self.register_buffer('_registered_keys', torch.tensor([], dtype=torch.long))
        
        # Embedding table - start empty
        self.embedding_table = nn.Embedding(0, embedding_dim)  # Empty embedding table initially
        
        # Track which keys we've seen
        self._key_to_index: Dict[int, int] = {}
        self._next_index = 0

    def _init_vector(self) -> torch.Tensor:
        return torch.nn.init.normal_(
            torch.empty(self.embedding_dim), 
            mean=self.init_mean, 
            std=self.init_std
        )

    def _expand_embedding_table(self, new_keys: List[int]) -> None:
        """Expand the embedding table to accommodate new keys"""
        num_new_keys = len(new_keys)
        if num_new_keys == 0:
            return
            
        # Create new embedding vectors
        new_embeddings = torch.stack([self._init_vector() for _ in range(num_new_keys)])
        
        # Expand the embedding table
        old_weight = self.embedding_table.weight.data
        new_weight = torch.cat([old_weight, new_embeddings], dim=0)
        
        # Create new embedding layer with expanded size
        new_embedding_table = nn.Embedding(
            new_weight.size(0), 
            self.embedding_dim
        )
        new_embedding_table.weight.data = new_weight
        self.embedding_table = new_embedding_table
        
        # Update key-to-index mapping
        for key in new_keys:
            self._key_to_index[key] = self._next_index
            self._next_index += 1
            
        # Update registered keys buffer
        new_registered_keys = torch.tensor(list(self._key_to_index.keys()), dtype=torch.long)
        self.register_buffer('_registered_keys', new_registered_keys)

    def _get_indices_for_keys(self, keys: torch.Tensor) -> torch.Tensor:
        """Convert keys to internal indices"""
        # Find new keys that need to be registered
        unique_keys = torch.unique(keys).cpu().tolist()
        new_keys = [k for k in unique_keys if k not in self._key_to_index]
        
        # Expand embedding table if needed
        if new_keys:
            self._expand_embedding_table(new_keys)
            
        # Map keys to indices
        indices = torch.tensor([self._key_to_index[k.item()] for k in keys.flatten()], 
                              device=keys.device)
        return indices.view(keys.shape)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Convert external keys to internal indices
        internal_indices = self._get_indices_for_keys(indices)
        
        # Use standard embedding lookup
        embeddings = self.embedding_table(internal_indices)
        
        return embeddings

    def _debug_print(self, key: int) -> None:
        if key in self._key_to_index:
            index = self._key_to_index[key]
            print(f'{key} exists in embedding', self.embedding_table.weight[index].tolist())
        else:
            print(f'{key} does not exist in embedding')