import sys 
import torch 
import torch.nn as nn
from kv_embedding import KVEmbedding
from typing import List, Dict, Tuple


class KVAdamOptimizer:
    def __init__(self, kv_embedding: KVEmbedding, **kwargs):
        self.kv_embedding = kv_embedding
        self.learning_rate = kwargs.get('lr', 0.001)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.999)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        self.weight_decay = kwargs.get('weight_decay', 0.0)
        # State for Adam optimizer
        # self._momentum_state: Dict[int, Tuple[torch.Tensor, torch.Tensor, int]] = {}
    
    def step(self):
        """Perform one optimization step"""
        _grad_accumulator = self.kv_embedding._grad_accumulator
        _momentum_state = self.kv_embedding._momentum_state

        for key in _grad_accumulator.keys():
            grad = _grad_accumulator[key]
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * self.kv_embedding.embedding[key]
            
            if key not in _momentum_state:
                _momentum_state[key] = (
                    torch.zeros_like(grad),
                    torch.zeros_like(grad),
                    0
                )
            first_momentum, second_momentum, time_step = self.kv_embedding._momentum_state[key]
            # Update time step
            time_step += 1

            # Update biased first moment estimate
            first_momentum = self.beta1 * first_momentum + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            second_momentum = self.beta2 * second_momentum + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = first_momentum / (1 - self.beta1 ** time_step)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = second_momentum / (1 - self.beta2 ** time_step)

            self.kv_embedding.embedding[key] -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)

            self.kv_embedding._momentum_state[key] = (first_momentum, second_momentum, time_step)

        self.kv_embedding._clear_gradients()
    
    def zero_grad(self):
        self.kv_embedding._clear_gradients()