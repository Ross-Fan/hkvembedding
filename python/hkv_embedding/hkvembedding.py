import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from . import hkv_core


class GradientBuffer:
    """
    GPU-backed gradient buffer using HKV for scalable gradient accumulation.
    Stores gradients directly in GPU memory via HKV hash table.
    """
    
    def __init__(self, embedding_dim: int, max_capacity: int, max_hbm_gb: int = 1):
        """
        Initialize GPU-backed gradient buffer.
        
        Args:
            embedding_dim: Dimension of embeddings (and gradients)
            max_capacity: Maximum number of unique keys to track
            max_hbm_gb: Maximum HBM for gradient storage
        """
        self.embedding_dim = embedding_dim
        # Store gradients: each entry is [grad (dim), count (1)] = dim + 1
        self.grad_dim = embedding_dim + 1  # extra dimension for count
        
        try:
            self.grad_table = hkv_core.HashTable(
                max_capacity // 10,  # init capacity
                max_capacity,
                self.grad_dim,
                max_hbm_gb
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create gradient buffer: {e}")
        
        self._pending_keys = set()  # Track which keys have gradients
    
    def accumulate(self, keys: np.ndarray, gradients: np.ndarray):
        """
        Accumulate gradients for given keys on GPU.
        
        Args:
            keys: uint64 array of keys
            gradients: float32 array of gradients [N, embedding_dim]
        """
        if len(keys) == 0:
            return
        
        batch_size = len(keys)
        
        # Get existing gradients (or zeros for new keys)
        existing_data, _ = self.grad_table.find_or_insert(keys)
        existing_data = existing_data.reshape(batch_size, self.grad_dim)
        
        # Split into gradients and counts
        existing_grads = existing_data[:, :self.embedding_dim]
        existing_counts = existing_data[:, -1:]
        
        # Accumulate new gradients and increment counts
        new_grads = existing_grads + gradients.reshape(batch_size, self.embedding_dim)
        new_counts = existing_counts + 1.0
        
        # Combine and update
        new_data = np.concatenate([new_grads, new_counts], axis=1).astype(np.float32)
        self.grad_table.insert_or_assign(keys, new_data.flatten())
        
        # Track keys
        self._pending_keys.update(keys.tolist())
    
    def get_averaged_gradients(self, keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get averaged gradients for keys.
        
        Returns:
            Tuple of (averaged_gradients, valid_mask)
        """
        if len(keys) == 0:
            return np.array([]).reshape(0, self.embedding_dim), np.array([], dtype=bool)
        
        batch_size = len(keys)
        data, found = self.grad_table.find(keys)
        data = data.reshape(batch_size, self.grad_dim)
        
        grads = data[:, :self.embedding_dim]
        counts = data[:, -1:].clip(min=1.0)  # Avoid division by zero
        
        averaged = grads / counts
        valid = found & (data[:, -1] > 0)
        
        return averaged.astype(np.float32), valid
    
    def get_all_pending_keys(self) -> np.ndarray:
        """Get all keys with pending gradients."""
        return np.array(list(self._pending_keys), dtype=np.uint64)
    
    def clear(self):
        """Clear all accumulated gradients."""
        self.grad_table.clear()
        self._pending_keys.clear()
    
    def size(self) -> int:
        """Get number of keys with gradients."""
        return len(self._pending_keys)


class AdamStateBuffer:
    """
    GPU-backed Adam optimizer state buffer using HKV.
    Stores m (first moment) and v (second moment) in HKV tables.
    """
    
    def __init__(self, embedding_dim: int, max_capacity: int, max_hbm_gb: int = 2):
        """
        Initialize Adam state buffer.
        
        Args:
            embedding_dim: Dimension of embeddings
            max_capacity: Maximum number of unique keys
            max_hbm_gb: Maximum HBM for state storage
        """
        self.embedding_dim = embedding_dim
        
        try:
            # Table for first moment (m)
            self.m_table = hkv_core.HashTable(
                max_capacity // 10,
                max_capacity,
                embedding_dim,
                max_hbm_gb // 2 or 1
            )
            
            # Table for second moment (v)
            self.v_table = hkv_core.HashTable(
                max_capacity // 10,
                max_capacity,
                embedding_dim,
                max_hbm_gb // 2 or 1
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create Adam state buffer: {e}")
    
    def get_states(self, keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get m and v states for keys (creates zeros for new keys).
        
        Returns:
            Tuple of (m_states, v_states) both shaped [N, embedding_dim]
        """
        if len(keys) == 0:
            return (np.array([]).reshape(0, self.embedding_dim),
                    np.array([]).reshape(0, self.embedding_dim))
        
        batch_size = len(keys)
        # Retrieve data and found flags, ensure new keys get zero initialization
        m_data, m_found = self.m_table.find_or_insert(keys)
        v_data, v_found = self.v_table.find_or_insert(keys)

        # Reshape arrays
        m_arr = m_data.reshape(batch_size, self.embedding_dim).astype(np.float32)
        v_arr = v_data.reshape(batch_size, self.embedding_dim).astype(np.float32)

        # For keys not previously present, ensure moments are zero
        if not np.all(m_found):
            missing_idx = np.where(~m_found)[0]
            if missing_idx.size > 0:
                m_arr[missing_idx, :] = 0.0

        if not np.all(v_found):
            missing_idx = np.where(~v_found)[0]
            if missing_idx.size > 0:
                v_arr[missing_idx, :] = 0.0

        # Guard against NaNs or negative values
        m_arr = np.nan_to_num(m_arr, nan=0.0, posinf=0.0, neginf=0.0)
        v_arr = np.nan_to_num(v_arr, nan=0.0, posinf=0.0, neginf=0.0)

        return (m_arr, v_arr)
    
    def update_states(self, keys: np.ndarray, m_new: np.ndarray, v_new: np.ndarray):
        """Update m and v states for keys."""
        if len(keys) == 0:
            return
        
        self.m_table.insert_or_assign(keys, m_new.flatten().astype(np.float32))
        self.v_table.insert_or_assign(keys, v_new.flatten().astype(np.float32))
    
    def clear(self):
        """Clear all states."""
        self.m_table.clear()
        self.v_table.clear()


class HKVEmbeddingFunction(torch.autograd.Function):
    """
    HKV Embedding custom autograd function.
    Supports forward and backward propagation with GPU-backed gradient accumulation.
    
    Note: This function uses a dummy tensor to enable gradient flow through
    the autograd graph, while actual gradient accumulation happens in the
    backward pass via the embedding layer's gradient buffer.
    """
    
    @staticmethod
    def forward(ctx, dummy_input, indices, embedding_layer):
        """
        Forward pass.
        
        Args:
            ctx: autograd context
            dummy_input: A dummy tensor with requires_grad=True to enable gradient flow
            indices: input index tensor (integer IDs)
            embedding_layer: HierarchicalHashEmbedding instance
        
        Returns:
            embeddings: embedding tensor
        """

        unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
        # 保存引用用于反向传播
        ctx.save_for_backward(indices, unique_indices, inverse_indices)
        ctx.embedding_layer = embedding_layer

        # 获取嵌入向量（不在autograd下执行HKV查找）
        # with torch.no_grad():
        #     embeddings = embedding_layer._forward_impl(indices)
        # 关键修改：使用已有的_forward_impl方法处理初始化逻辑
        with torch.no_grad():
            unique_embeddings = embedding_layer._forward_impl(unique_indices)
        
        # Tie the returned tensor to the dummy_input so that autograd will
        # invoke this Function.backward. Adding a zero-valued scalar that
        # depends on dummy_input creates a dependency without changing values.
        # print("unique_embeddings:", unique_embeddings)
        
        embeddings = unique_embeddings[inverse_indices] + (dummy_input.sum() * 0.0)

        return embeddings
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass - accumulates gradients in GPU-backed buffer.
        """
        print("grad_output:", grad_output)
        print("grad_output shape:", grad_output.shape)
        indices, unique_indices, inverse_indices = ctx.saved_tensors
        embedding_layer = ctx.embedding_layer
        
        # 累积梯度到GPU缓冲区
        if grad_output is not None:
            embedding_layer._accumulate_gradients(indices, grad_output)
        
        # indices不需要梯度，embedding_layer也不需要
        return embedding_layer._grad_dummy, None, None


class HierarchicalHashEmbedding(nn.Module):
    """
    PyTorch embedding layer based on HierarchicalKV with automatic differentiation.
    Designed for large-scale recommendation systems with billions of unique IDs.
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 max_capacity: int = 1000000000,
                 init_capacity: int = 10000000,
                 max_hbm_gb: int = 8,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32,
                 init_std: float = None,
                 learning_rate: float = 0.01,
                 weight_decay: float = 0.0,
                 grad_buffer_hbm_gb: int = 1,
                 debug_print: bool = False):
        """
        Initialize HierarchicalHashEmbedding.
        
        Args:
            embedding_dim: Embedding vector dimension
            max_capacity: Maximum capacity
            init_capacity: Initial capacity
            max_hbm_gb: Maximum GPU memory usage (GB) for embeddings
            device: Device type
            dtype: Data type
            init_std: Initialization standard deviation
            learning_rate: Learning rate for built-in optimizer
            weight_decay: Weight decay
            grad_buffer_hbm_gb: GPU memory for gradient buffer (GB)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_capacity = max_capacity
        self.init_capacity = init_capacity
        self.device = device
        self.dtype = dtype
        self.init_std = init_std or 0.1 * (2.0 / embedding_dim)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self._debug_print = debug_print  # 设置调试打印标志

        # Create HKV hash table for embeddings
        try:
            self.hashtable = hkv_core.HashTable(
                init_capacity, max_capacity, embedding_dim, max_hbm_gb
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create HashTable: {e}")
        
        # GPU-backed gradient buffer (replaces Python dict)
        self.grad_buffer = GradientBuffer(
            embedding_dim=embedding_dim,
            max_capacity=max_capacity,
            max_hbm_gb=grad_buffer_hbm_gb
        )
        
        # Statistics
        self.hit_count = 0
        self.miss_count = 0
        self.total_queries = 0
        
        # Track keys seen in current forward pass (for initialization)
        self._current_batch_new_keys = None
        
        # Register dummy parameter for gradient flow in autograd.Function
        # This tensor enables the backward pass to be triggered
        self._grad_dummy = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=True)
    
    def forward(self, indices):
        """
        Forward pass with automatic differentiation.
        
        Args:
            indices: Input index tensor
            
        Returns:
            embeddings: Embedding tensor
        """
        if indices.numel() == 0:
            return torch.empty(0, self.embedding_dim, dtype=self.dtype, device=self.device)

        original_shape = indices.shape
        indices_flat = indices.flatten()

        # 确保dummy参数在正确的设备上
        if self._grad_dummy.device != self.device:
            self._grad_dummy = self._grad_dummy.to(self.device)
        
        # 使用自定义autograd函数，传入dummy参数以启用梯度流
        embeddings = HKVEmbeddingFunction.apply(self._grad_dummy, indices_flat, self)
            
        # 恢复原始形状
        if len(original_shape) == 1:
            return embeddings
        else:
            return embeddings.view(*original_shape, self.embedding_dim)
        
    def _forward_impl(self, indices):
        """
        Actual forward implementation.
        
        Args:
            indices: Flattened index tensor
            
        Returns:
            embeddings: Embedding tensor
        """
        # Convert to numpy
        indices_np = indices.cpu().numpy().astype(np.uint64)
        
        # First, check which keys exist
        try:
            embeddings_np, found_flags = self.hashtable.find(indices_np)
        except Exception as e:
            raise RuntimeError(f"HKV find failed: {e}")
        
        # Convert to PyTorch tensor
        embeddings = torch.from_numpy(embeddings_np).to(
            device=self.device, dtype=self.dtype
        )
        
        # Handle new keys (not found)
        found_tensor = torch.from_numpy(found_flags)
        new_mask = ~found_tensor
        
        if new_mask.any():
            num_new = new_mask.sum().item()
            
            # Random initialization for new embeddings
            new_embeddings = torch.randn(
                num_new, self.embedding_dim,
                device=self.device, dtype=self.dtype
            ) * self.init_std
            
            embeddings[new_mask] = new_embeddings
            
            # Insert new embeddings into hash table
            new_keys = indices_np[new_mask.numpy()]
            self._update_embeddings_impl(
                torch.from_numpy(new_keys), 
                new_embeddings
            )
            
            self.miss_count += num_new
        
        # Update statistics
        self.hit_count += found_flags.sum()
        self.total_queries += len(found_flags)
        
        return embeddings
    
    def _accumulate_gradients(self, indices: torch.Tensor, grad_output: torch.Tensor):
        """
        Accumulate gradients using GPU-backed buffer.
        
        Args:
            indices: Index tensor
            grad_output: Output gradients
        """
        if grad_output is None or indices.numel() == 0:
            return
        
        # 确保张量在正确的设备上
        if indices.device != self.device:
            indices = indices.to(self.device)
        if grad_output.device != self.device:
            grad_output = grad_output.to(self.device)

        # 打印梯度信息用于调试
        if hasattr(self, '_debug_print') and self._debug_print:
            print(f"Indices shape: {indices.shape}, Grad output shape: {grad_output.shape}")
            print(f"Grad output stats - Min: {grad_output.min().item():.6f}, "
                f"Max: {grad_output.max().item():.6f}, "
                f"Mean: {grad_output.mean().item():.6f}, "
                f"Std: {grad_output.std().item():.6f}")
            
            # 打印前几个梯度值的样本
            if grad_output.numel() > 0:
                sample_grads = grad_output.flatten()[:10]
                print(f"Sample gradients: {sample_grads.tolist()}")
            
            # 检查是否有异常值
            if torch.isnan(grad_output).any():
                print("WARNING: NaN detected in gradients!")
            if torch.isinf(grad_output).any():
                print("WARNING: Inf detected in gradients!")

        # Convert to numpy for HKV
        indices_np = indices.cpu().numpy().astype(np.uint64)
        grad_np = grad_output.detach().cpu().numpy().astype(np.float32)
        
        # Accumulate to GPU buffer
        self.grad_buffer.accumulate(indices_np, grad_np)
    
    def step(self):
        """
        Execute one gradient update step (like optimizer.step()).
        Uses SGD with optional weight decay.
        """
        pending_keys = self.grad_buffer.get_all_pending_keys()
        
        if len(pending_keys) == 0:
            return
        
        # Get averaged gradients
        avg_grads, valid_mask = self.grad_buffer.get_averaged_gradients(pending_keys)
        
        if not valid_mask.any():
            self.grad_buffer.clear()
            return
        
        # Filter to valid keys
        valid_keys = pending_keys[valid_mask]
        valid_grads = avg_grads[valid_mask]
        
        # Get current embeddings
        current_embeddings, found = self.hashtable.find(valid_keys)
        current_embeddings = current_embeddings.reshape(-1, self.embedding_dim)
        
        # Apply weight decay if specified
        if self.weight_decay > 0:
            valid_grads = valid_grads + self.weight_decay * current_embeddings
        
        # SGD update: embedding = embedding - lr * grad
        updated_embeddings = current_embeddings - self.learning_rate * valid_grads
        
        # Update hash table
        self.hashtable.insert_or_assign(
            valid_keys, 
            updated_embeddings.flatten().astype(np.float32)
        )
        
        # Clear gradient buffer
        self.grad_buffer.clear()
    
    def apply_updates(self, keys: np.ndarray, updates: np.ndarray):
        """
        Apply precomputed updates to embeddings (used by external optimizers).
        
        Args:
            keys: uint64 array of keys
            updates: float32 array of update values [N, embedding_dim]
        """
        if len(keys) == 0:
            return
        
        # Get current embeddings
        current_embeddings, found = self.hashtable.find(keys)
        current_embeddings = current_embeddings.reshape(-1, self.embedding_dim)
        
        # Apply updates: embedding = embedding - update
        updated_embeddings = current_embeddings - updates.reshape(-1, self.embedding_dim)
        
        # Update hash table
        self.hashtable.insert_or_assign(
            keys,
            updated_embeddings.flatten().astype(np.float32)
        )
    
    def _update_embeddings_impl(self, indices: torch.Tensor, embeddings: torch.Tensor):
        """Update embeddings in hash table (internal implementation)."""
        if indices.numel() == 0:
            return
            
        indices_np = indices.cpu().numpy().astype(np.uint64)
        embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)
        
        try:
            self.hashtable.insert_or_assign(indices_np, embeddings_np.flatten())
        except Exception as e:
            raise RuntimeError(f"HKV insert_or_assign failed: {e}")
    
    def zero_grad(self):
        """Clear gradients (like optimizer.zero_grad())."""
        self.grad_buffer.clear()
    
    def get_pending_gradient_count(self) -> int:
        """Get number of keys with pending gradients."""
        return self.grad_buffer.size()
    
    def get_embedding(self, key: int) -> Optional[torch.Tensor]:
        """Get embedding for a single key."""
        try:
            keys_np = np.array([key], dtype=np.uint64)
            embeddings_np, found = self.hashtable.find(keys_np)
            if found[0]:
                return torch.from_numpy(embeddings_np.reshape(self.embedding_dim))
            return None
        except:
            return None
    
    def get_embeddings(self, keys: Union[List[int], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Get embeddings for multiple keys."""
        if isinstance(keys, torch.Tensor):
            keys_np = keys.cpu().numpy().astype(np.uint64)
        elif isinstance(keys, list):
            keys_np = np.array(keys, dtype=np.uint64)
        else:
            keys_np = keys.astype(np.uint64)
        
        embeddings_np, _ = self.hashtable.find(keys_np)
        return torch.from_numpy(embeddings_np.reshape(-1, self.embedding_dim)).to(
            device=self.device, dtype=self.dtype
        )
    
    def parameters(self):
        """Return trainable parameters (for PyTorch optimizer compatibility)."""
        return [self._grad_dummy]
    
    def named_parameters(self, prefix='', recurse=True):
        """Return named parameters."""
        yield prefix + ('.' if prefix else '') + '_grad_dummy', self._grad_dummy
    
    def state_dict(self):
        """Save state dict."""
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
        """Load state dict."""
        self.embedding_dim = state_dict.get('embedding_dim', self.embedding_dim)
        self.max_capacity = state_dict.get('max_capacity', self.max_capacity)
        self.init_capacity = state_dict.get('init_capacity', self.init_capacity)
        self.learning_rate = state_dict.get('learning_rate', self.learning_rate)
        self.weight_decay = state_dict.get('weight_decay', self.weight_decay)
        self.hit_count = state_dict.get('hit_count', 0)
        self.miss_count = state_dict.get('miss_count', 0)
        self.total_queries = state_dict.get('total_queries', 0)
    
    def export_embeddings(self, batch_size: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export all embeddings from the hash table.
        
        Args:
            batch_size: Batch size for export
            
        Returns:
            Tuple of (keys, embeddings)
        """
        # Note: This requires implementing export in C++ layer
        # For now, raise NotImplementedError
        raise NotImplementedError(
            "Export functionality requires C++ implementation. "
            "Use hashtable's export_batch if available."
        )
    
    def get_statistics(self) -> Dict:
        """Get statistics."""
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
            'pending_gradients': self.get_pending_gradient_count(),
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


class MultiTableHKVEmbedding(nn.Module):
    """
    Multiple HKV embedding tables for different feature fields.
    Common in recommendation systems with multiple categorical features.
    """
    
    def __init__(self,
                 num_tables: int,
                 embedding_dim: int,
                 max_capacity_per_table: int = 1000000,
                 init_capacity_per_table: int = 100000,
                 max_hbm_gb_per_table: int = 4,
                 learning_rate: float = 0.001,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32,
                 shared_optimizer: bool = True,
                 debug_print: bool = False):
        """
        Initialize multiple embedding tables.
        
        Args:
            num_tables: Number of embedding tables
            embedding_dim: Embedding dimension (same for all tables)
            max_capacity_per_table: Maximum capacity per table
            init_capacity_per_table: Initial capacity per table
            max_hbm_gb_per_table: HBM per table
            device: Device
            dtype: Data type
            shared_optimizer: Whether to use shared optimizer settings
        """
        super().__init__()
        
        self.num_tables = num_tables
        self.embedding_dim = embedding_dim
        self.debug_print = debug_print
        
        # Create embedding tables
        self.tables = nn.ModuleList([
            HierarchicalHashEmbedding(
                embedding_dim=embedding_dim,
                max_capacity=max_capacity_per_table,
                init_capacity=init_capacity_per_table,
                max_hbm_gb=max_hbm_gb_per_table,
                learning_rate=learning_rate,
                device=device,
                dtype=dtype,
                debug_print=self.debug_print
            )
            for _ in range(num_tables)
        ])
    
    def forward(self, indices_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass for all tables.
        
        Args:
            indices_list: List of index tensors, one per table
            
        Returns:
            List of embedding tensors
        """
        assert len(indices_list) == self.num_tables, \
            f"Expected {self.num_tables} index tensors, got {len(indices_list)}"
        
        return [table(indices) for table, indices in zip(self.tables, indices_list)]
    
    def zero_grad(self):
        """Clear all gradients."""
        for table in self.tables:
            table.zero_grad()
    
    def step(self):
        """Update all tables."""
        for table in self.tables:
            table.step()
    
    def get_all_tables(self) -> List[HierarchicalHashEmbedding]:
        """Get all embedding tables."""
        return list(self.tables)
