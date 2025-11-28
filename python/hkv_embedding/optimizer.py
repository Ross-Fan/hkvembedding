import torch
import numpy as np
from typing import List, Dict, Any, Union
from .hkvembedding import HierarchicalHashEmbedding, AdamStateBuffer


class HKVOptimizer:
    """
    HKV Embedding SGD optimizer.
    Uses GPU-backed gradient storage for scalability.
    """
    
    def __init__(self, 
                 hkv_embeddings: Union[HierarchicalHashEmbedding, List[HierarchicalHashEmbedding]], 
                 lr: float = 0.01, 
                 weight_decay: float = 0.0):
        """
        Initialize optimizer.
        
        Args:
            hkv_embeddings: HierarchicalHashEmbedding instance(s)
            lr: Learning rate
            weight_decay: Weight decay (L2 regularization)
        """
        if isinstance(hkv_embeddings, list):
            self.hkv_embeddings = hkv_embeddings
        else:
            self.hkv_embeddings = [hkv_embeddings]
        
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Update all embeddings' learning rate
        for embedding in self.hkv_embeddings:
            embedding.learning_rate = lr
            embedding.weight_decay = weight_decay
    
    def step(self):
        """Execute one optimization step."""
        for embedding in self.hkv_embeddings:
            embedding.step()
    
    def zero_grad(self):
        """Clear gradients."""
        for embedding in self.hkv_embeddings:
            embedding.zero_grad()
    
    def state_dict(self) -> Dict[str, Any]:
        """Save optimizer state."""
        return {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state."""
        self.lr = state_dict['lr']
        self.weight_decay = state_dict['weight_decay']
        
        for embedding in self.hkv_embeddings:
            embedding.learning_rate = self.lr
            embedding.weight_decay = self.weight_decay


class HKVAdamOptimizer:
    """
    HKV Embedding Adam optimizer.
    Uses GPU-backed HKV tables for momentum states (m, v) to handle
    billions of unique IDs without Python dict memory issues.
    """
    
    def __init__(self, 
                 hkv_embeddings: Union[HierarchicalHashEmbedding, List[HierarchicalHashEmbedding]], 
                 lr: float = 0.001, 
                 betas: tuple = (0.9, 0.999), 
                 eps: float = 1e-8, 
                 weight_decay: float = 0.0,
                 state_hbm_gb_per_embedding: int = 2,
                 max_grad_norm: float = 10.0):
        """
        Initialize Adam optimizer with GPU-backed state storage.
        
        Args:
            hkv_embeddings: HierarchicalHashEmbedding instance(s)
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay (L2 regularization)
            state_hbm_gb_per_embedding: HBM for Adam states per embedding
        """
        if isinstance(hkv_embeddings, list):
            self.hkv_embeddings = hkv_embeddings
        else:
            self.hkv_embeddings = [hkv_embeddings]
        
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        
        # Create GPU-backed Adam state buffers for each embedding
        # Uses HKV tables instead of Python dicts for scalability
        self.state_buffers: Dict[int, AdamStateBuffer] = {}
        
        for i, embedding in enumerate(self.hkv_embeddings):
            self.state_buffers[i] = AdamStateBuffer(
                embedding_dim=embedding.embedding_dim,
                max_capacity=embedding.max_capacity,
                max_hbm_gb=state_hbm_gb_per_embedding
            )
        # Gradient clipping threshold (L2 norm per-key)
        self.max_grad_norm = max_grad_norm
    
    def step(self):
        """Execute Adam optimization step using GPU-backed state."""
        self.step_count += 1
        beta1, beta2 = self.betas
        
        for idx, embedding in enumerate(self.hkv_embeddings):
            # Get pending gradient keys
            pending_keys = embedding.grad_buffer.get_all_pending_keys()
            
            if len(pending_keys) == 0:
                continue
            
            # Get averaged gradients
            avg_grads, valid_mask = embedding.grad_buffer.get_averaged_gradients(pending_keys)
            
            if not valid_mask.any():
                embedding.zero_grad()
                continue
            
            # Filter to valid keys
            valid_keys = pending_keys[valid_mask]
            valid_grads = avg_grads[valid_mask]
            batch_size = len(valid_keys)
            
            # Get current embeddings
            current_embeddings, _ = embedding.hashtable.find(valid_keys)
            current_embeddings = current_embeddings.reshape(batch_size, embedding.embedding_dim)
            
            # Apply weight decay to gradients (AdamW style)
            if self.weight_decay > 0:
                valid_grads = valid_grads + self.weight_decay * current_embeddings
            
            # Get Adam states (m, v) from GPU-backed buffer
            state_buffer = self.state_buffers[idx]
            m_prev, v_prev = state_buffer.get_states(valid_keys)
            # Ensure numerical sanity
            m_prev = np.nan_to_num(m_prev, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            v_prev = np.nan_to_num(v_prev, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            valid_grads = np.nan_to_num(valid_grads, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            # Update biased first moment estimate
            m_new = beta1 * m_prev + (1 - beta1) * valid_grads

            # Update biased second raw moment estimate
            v_new = beta2 * v_prev + (1 - beta2) * (valid_grads ** 2)
            
            # Store updated states
            state_buffer.update_states(valid_keys, m_new, v_new)
            
            # Optional per-key gradient clipping to avoid exploding updates
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                norms = np.linalg.norm(valid_grads, axis=1, keepdims=True)
                scale = np.minimum(1.0, self.max_grad_norm / (norms + 1e-6))
                valid_grads = valid_grads * scale

            # Compute bias-corrected estimates
            m_hat = m_new / (1 - beta1 ** self.step_count)
            v_hat = v_new / (1 - beta2 ** self.step_count)

            # Numeric guards: ensure v_hat non-negative to avoid sqrt of negative
            v_hat = np.maximum(v_hat, 0.0)

            # Compute update safely
            denom = np.sqrt(v_hat) + self.eps
            # Avoid division by zero
            denom = np.where(denom == 0.0, self.eps, denom)
            update = self.lr * m_hat / denom
            
            # Apply update: embedding = embedding - update
            updated_embeddings = current_embeddings - update
            
            # Update hash table
            embedding.hashtable.insert_or_assign(
                valid_keys,
                updated_embeddings.flatten().astype(np.float32)
            )
            
            # Clear gradients
            embedding.zero_grad()
    
    def zero_grad(self):
        """Clear gradients for all embeddings."""
        for embedding in self.hkv_embeddings:
            embedding.zero_grad()
    
    def state_dict(self) -> Dict[str, Any]:
        """Save optimizer state."""
        return {
            'lr': self.lr,
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'step_count': self.step_count,
            # Note: Adam states (m, v) are stored in HKV tables
            # Full state saving would require exporting HKV tables
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state."""
        self.lr = state_dict['lr']
        self.betas = state_dict['betas']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.step_count = state_dict.get('step_count', 0)
    
    def reset_state(self):
        """Reset Adam states (m, v) for all embeddings."""
        for state_buffer in self.state_buffers.values():
            state_buffer.clear()
        self.step_count = 0


class HKVSparseAdam:
    """
    Sparse Adam optimizer that only updates embeddings seen in current batch.
    More memory efficient for very sparse access patterns.
    """
    
    def __init__(self,
                 hkv_embeddings: Union[HierarchicalHashEmbedding, List[HierarchicalHashEmbedding]],
                 lr: float = 0.001,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 state_hbm_gb_per_embedding: int = 2,
                 amsgrad: bool = False):
        """
        Initialize Sparse Adam optimizer.
        
        Args:
            hkv_embeddings: HierarchicalHashEmbedding instance(s)
            lr: Learning rate
            betas: Coefficients for running averages
            eps: Numerical stability term
            weight_decay: Weight decay
            state_hbm_gb_per_embedding: HBM for states per embedding
            amsgrad: Whether to use AMSGrad variant
        """
        if isinstance(hkv_embeddings, list):
            self.hkv_embeddings = hkv_embeddings
        else:
            self.hkv_embeddings = [hkv_embeddings]
        
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # Per-key step counts stored in HKV (for proper bias correction)
        self.step_counts: Dict[int, Dict[int, int]] = {}
        
        # State buffers
        self.state_buffers: Dict[int, AdamStateBuffer] = {}
        
        for i, embedding in enumerate(self.hkv_embeddings):
            self.state_buffers[i] = AdamStateBuffer(
                embedding_dim=embedding.embedding_dim,
                max_capacity=embedding.max_capacity,
                max_hbm_gb=state_hbm_gb_per_embedding
            )
            self.step_counts[i] = {}
    
    def step(self):
        """Execute Sparse Adam step with per-key step counting."""
        beta1, beta2 = self.betas
        
        for idx, embedding in enumerate(self.hkv_embeddings):
            pending_keys = embedding.grad_buffer.get_all_pending_keys()
            
            if len(pending_keys) == 0:
                continue
            
            avg_grads, valid_mask = embedding.grad_buffer.get_averaged_gradients(pending_keys)
            
            if not valid_mask.any():
                embedding.zero_grad()
                continue
            
            valid_keys = pending_keys[valid_mask]
            valid_grads = avg_grads[valid_mask]
            batch_size = len(valid_keys)
            
            # Get current embeddings
            current_embeddings, _ = embedding.hashtable.find(valid_keys)
            current_embeddings = current_embeddings.reshape(batch_size, embedding.embedding_dim)
            
            # Get/create step counts for each key
            step_counts_dict = self.step_counts[idx]
            key_steps = np.array([
                step_counts_dict.get(int(k), 0) + 1 
                for k in valid_keys
            ], dtype=np.float32).reshape(-1, 1)
            
            # Update step counts
            for k in valid_keys:
                k_int = int(k)
                step_counts_dict[k_int] = step_counts_dict.get(k_int, 0) + 1
            
            # Apply weight decay
            if self.weight_decay > 0:
                valid_grads = valid_grads + self.weight_decay * current_embeddings
            
            # Get Adam states
            state_buffer = self.state_buffers[idx]
            m_prev, v_prev = state_buffer.get_states(valid_keys)
            
            # Update moments
            m_new = beta1 * m_prev + (1 - beta1) * valid_grads
            v_new = beta2 * v_prev + (1 - beta2) * (valid_grads ** 2)
            
            # Store updated states
            state_buffer.update_states(valid_keys, m_new, v_new)
            
            # Bias correction with per-key step counts
            m_hat = m_new / (1 - np.power(beta1, key_steps))
            v_hat = v_new / (1 - np.power(beta2, key_steps))
            
            # Compute update
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            # Apply update
            updated_embeddings = current_embeddings - update
            
            embedding.hashtable.insert_or_assign(
                valid_keys,
                updated_embeddings.flatten().astype(np.float32)
            )
            
            embedding.zero_grad()
    
    def zero_grad(self):
        """Clear gradients."""
        for embedding in self.hkv_embeddings:
            embedding.zero_grad()
    
    def state_dict(self) -> Dict[str, Any]:
        """Save state."""
        return {
            'lr': self.lr,
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'step_counts': {k: dict(v) for k, v in self.step_counts.items()},
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state."""
        self.lr = state_dict['lr']
        self.betas = state_dict['betas']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.step_counts = {
            int(k): v for k, v in state_dict.get('step_counts', {}).items()
        }


class HKVAdagrad:
    """
    Adagrad optimizer for HKV embeddings.
    Particularly suitable for sparse features in recommendation systems.
    """
    
    def __init__(self,
                 hkv_embeddings: Union[HierarchicalHashEmbedding, List[HierarchicalHashEmbedding]],
                 lr: float = 0.01,
                 lr_decay: float = 0.0,
                 initial_accumulator_value: float = 0.0,
                 eps: float = 1e-10,
                 weight_decay: float = 0.0,
                 state_hbm_gb_per_embedding: int = 1):
        """
        Initialize Adagrad optimizer.
        
        Args:
            hkv_embeddings: HierarchicalHashEmbedding instance(s)
            lr: Learning rate
            lr_decay: Learning rate decay
            initial_accumulator_value: Initial value for sum of squared gradients
            eps: Numerical stability
            weight_decay: Weight decay
            state_hbm_gb_per_embedding: HBM for accumulator state
        """
        if isinstance(hkv_embeddings, list):
            self.hkv_embeddings = hkv_embeddings
        else:
            self.hkv_embeddings = [hkv_embeddings]
        
        self.lr = lr
        self.lr_decay = lr_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        
        # Sum of squared gradients stored in HKV tables
        from . import hkv_core
        self.accumulators: Dict[int, Any] = {}
        
        for i, embedding in enumerate(self.hkv_embeddings):
            try:
                self.accumulators[i] = hkv_core.HashTable(
                    embedding.init_capacity // 10,
                    embedding.max_capacity,
                    embedding.embedding_dim,
                    state_hbm_gb_per_embedding
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create Adagrad accumulator: {e}")
    
    def step(self):
        """Execute Adagrad step."""
        self.step_count += 1
        
        # Compute decayed learning rate
        lr = self.lr / (1 + (self.step_count - 1) * self.lr_decay)
        
        for idx, embedding in enumerate(self.hkv_embeddings):
            pending_keys = embedding.grad_buffer.get_all_pending_keys()
            
            if len(pending_keys) == 0:
                continue
            
            avg_grads, valid_mask = embedding.grad_buffer.get_averaged_gradients(pending_keys)
            
            if not valid_mask.any():
                embedding.zero_grad()
                continue
            
            valid_keys = pending_keys[valid_mask]
            valid_grads = avg_grads[valid_mask]
            batch_size = len(valid_keys)
            
            # Get current embeddings
            current_embeddings, _ = embedding.hashtable.find(valid_keys)
            current_embeddings = current_embeddings.reshape(batch_size, embedding.embedding_dim)
            
            # Apply weight decay
            if self.weight_decay > 0:
                valid_grads = valid_grads + self.weight_decay * current_embeddings
            
            # Get accumulated squared gradients
            accumulator = self.accumulators[idx]
            acc_data, _ = accumulator.find_or_insert(valid_keys)
            acc_prev = acc_data.reshape(batch_size, embedding.embedding_dim)
            
            # Initialize with initial_accumulator_value for new keys
            if self.initial_accumulator_value > 0:
                zero_mask = np.all(acc_prev == 0, axis=1)
                acc_prev[zero_mask] = self.initial_accumulator_value
            
            # Update accumulator: acc = acc + grad^2
            acc_new = acc_prev + valid_grads ** 2
            
            # Store updated accumulator
            accumulator.insert_or_assign(
                valid_keys,
                acc_new.flatten().astype(np.float32)
            )
            
            # Compute update: lr * grad / sqrt(acc + eps)
            update = lr * valid_grads / (np.sqrt(acc_new) + self.eps)
            
            # Apply update
            updated_embeddings = current_embeddings - update
            
            embedding.hashtable.insert_or_assign(
                valid_keys,
                updated_embeddings.flatten().astype(np.float32)
            )
            
            embedding.zero_grad()
    
    def zero_grad(self):
        """Clear gradients."""
        for embedding in self.hkv_embeddings:
            embedding.zero_grad()
    
    def state_dict(self) -> Dict[str, Any]:
        """Save state."""
        return {
            'lr': self.lr,
            'lr_decay': self.lr_decay,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'step_count': self.step_count,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state."""
        self.lr = state_dict['lr']
        self.lr_decay = state_dict['lr_decay']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.step_count = state_dict.get('step_count', 0)
