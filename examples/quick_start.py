import torch
import hkv_python_binding as hkv

# Create embedding layer
embedding = hkv.HierarchicalHashEmbedding(
    embedding_dim=64,
    max_capacity=2**32,
    init_capacity=100000,
    max_hbm_gb=2
)

# Use in PyTorch model
indices = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
embeddings = embedding(indices)
print(f"Embeddings shape: {embeddings.shape}")

# Get statistics
stats = embedding.get_statistics()
print(f"Hit rate: {stats['hit_rate']:.2%}")
