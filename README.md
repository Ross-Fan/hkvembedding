# HKV Embedding

High-performance GPU hashtable embeddings for PyTorch using [NVIDIA HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV).

Designed for large-scale recommendation systems with **billions of unique IDs**.

## Features

- **GPU-Native Storage**: Embeddings stored on GPU HBM (High Bandwidth Memory) via HierarchicalKV
- **Scalable Gradient Handling**: GPU-backed gradient buffers - no Python dict memory issues with billions of IDs
- **GPU-Backed Optimizer States**: Adam/Adagrad momentum states stored in HKV tables
- **PyTorch Integration**: Full autograd support with `torch.autograd.Function`
- **LRU Eviction**: Automatic eviction of least-recently-used embeddings when capacity is reached
- **Multiple Optimizers**: SGD, Adam, Sparse Adam, and Adagrad optimizers included

## Installation

```bash
# Build from source
./build_wheel.sh

# Install
pip install dist/hkv_embedding-*.whl
```

### Requirements

- CUDA 11.0+
- PyTorch 1.9+
- pybind11
- HierarchicalKV (included as submodule)

## Quick Start

```python
import torch
import hkv_embedding

# Create embedding layer for billion-scale IDs
embedding = hkv_embedding.HierarchicalHashEmbedding(
    embedding_dim=64,
    max_capacity=1_000_000_000,  # 1 billion IDs
    init_capacity=1_000_000,
    max_hbm_gb=16,  # 16GB GPU memory for embeddings
    grad_buffer_hbm_gb=2  # 2GB for gradient accumulation
)

# Create optimizer (GPU-backed Adam states)
optimizer = hkv_embedding.HKVAdamOptimizer(
    [embedding],
    lr=0.001,
    state_hbm_gb_per_embedding=4  # 4GB for Adam m/v states
)

# Use like any PyTorch embedding
indices = torch.randint(0, 1_000_000_000, (1024,), device='cuda')
embeddings = embedding(indices)

# Backward pass accumulates gradients to GPU buffer
loss = embeddings.mean()
loss.backward()

# Update embeddings
optimizer.step()
optimizer.zero_grad()
```

## Architecture

### The Problem with Traditional Approaches

Standard PyTorch embeddings and dictionary-based gradient storage don't scale:

```python
# ❌ Traditional approach - fails with billions of IDs
self.gradient_accumulator = {}  # Python dict grows unbounded
self.m_dict = {}  # Adam first moment - OOM with billions of keys
self.v_dict = {}  # Adam second moment - OOM with billions of keys
```

### HKV Embedding Solution

```python
# ✅ HKV approach - GPU-backed storage for everything
self.hashtable = hkv_core.HashTable(...)      # Embeddings in GPU HBM
self.grad_buffer = GradientBuffer(...)        # Gradients in GPU HKV table
self.adam_states = AdamStateBuffer(...)       # Adam m/v in GPU HKV tables
```

## Key Components

### `HierarchicalHashEmbedding`

Main embedding layer with autograd support:

```python
embedding = hkv_embedding.HierarchicalHashEmbedding(
    embedding_dim=64,           # Embedding vector dimension
    max_capacity=1_000_000_000, # Maximum unique IDs
    init_capacity=1_000_000,    # Initial allocation
    max_hbm_gb=16,              # GPU memory for embeddings
    learning_rate=0.01,         # For built-in SGD
    weight_decay=0.0,           # L2 regularization
    grad_buffer_hbm_gb=1        # GPU memory for gradients
)
```

### `MultiTableHKVEmbedding`

For multiple feature fields (common in RecSys):

```python
multi_embedding = hkv_embedding.MultiTableHKVEmbedding(
    num_tables=10,              # 10 feature fields
    embedding_dim=32,
    max_capacity_per_table=100_000_000,
    max_hbm_gb_per_table=2
)

# Forward pass
embeddings_list = multi_embedding([field1_ids, field2_ids, ...])
```

### Optimizers

All optimizers use GPU-backed state storage:

```python
# SGD (simplest)
optimizer = hkv_embedding.HKVOptimizer(embeddings, lr=0.01)

# Adam (recommended)
optimizer = hkv_embedding.HKVAdamOptimizer(
    embeddings,
    lr=0.001,
    betas=(0.9, 0.999),
    state_hbm_gb_per_embedding=2  # GPU memory for m/v states
)

# Adagrad (good for sparse features)
optimizer = hkv_embedding.HKVAdagrad(
    embeddings,
    lr=0.1,
    state_hbm_gb_per_embedding=1
)

# Sparse Adam (per-key step counting)
optimizer = hkv_embedding.HKVSparseAdam(
    embeddings,
    lr=0.001,
    state_hbm_gb_per_embedding=2
)
```

## Example Models

### DeepFM

```python
class DeepFMModel(nn.Module):
    def __init__(self, num_fields, embedding_dim):
        super().__init__()
        self.embeddings = hkv_embedding.MultiTableHKVEmbedding(
            num_tables=num_fields,
            embedding_dim=embedding_dim,
            max_capacity_per_table=100_000_000
        )
        self.mlp = nn.Sequential(...)
    
    def forward(self, sparse_features):
        emb_list = self.embeddings(sparse_features)
        # FM + Deep components
        ...
```

### Two-Tower Retrieval

```python
class TwoTowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_emb = hkv_embedding.HierarchicalHashEmbedding(
            embedding_dim=64,
            max_capacity=1_000_000_000  # 1B users
        )
        self.item_emb = hkv_embedding.HierarchicalHashEmbedding(
            embedding_dim=64,
            max_capacity=100_000_000   # 100M items
        )
        self.user_tower = nn.Sequential(...)
        self.item_tower = nn.Sequential(...)
```

## Memory Planning

For a model with 1 billion embeddings at 64 dimensions:

| Component | Size | Notes |
|-----------|------|-------|
| Embeddings | ~256 GB | 1B × 64 × 4 bytes |
| Gradients | Variable | Only active keys |
| Adam m | ~256 GB | Same as embeddings |
| Adam v | ~256 GB | Same as embeddings |

With HKV's HBM+HMEM hybrid mode, you can:
- Store hot embeddings on GPU HBM
- Overflow to host memory (HMEM)
- Use LRU eviction for capacity management

## Performance Tips

1. **Batch Size**: Use large batches (4096+) to amortize HKV lookup overhead
2. **Gradient Buffer**: Size based on unique IDs per batch × accumulation steps
3. **Adam States**: Will grow as unique IDs are seen; plan HBM accordingly
4. **Warm-up**: First few iterations may be slower as embeddings are initialized

## Comparison with Alternatives

| Feature | nn.Embedding | FBGEMM | HKV Embedding |
|---------|--------------|--------|---------------|
| Max IDs | ~100M | ~1B | Unlimited* |
| Storage | GPU only | CPU+GPU | HBM+HMEM |
| Eviction | No | Yes | LRU |
| Optimizer States | RAM | RAM | GPU HKV |
| PyTorch Integration | Native | Custom | Native |

*Limited by combined HBM+HMEM capacity

## API Reference

See [examples/](examples/) for detailed usage examples.

## License

Apache 2.0

## Acknowledgments

- [NVIDIA HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV) - GPU hash table implementation
- [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin) - Recommendation system framework
