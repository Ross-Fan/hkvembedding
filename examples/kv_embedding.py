import sys
import torch 
import torch.nn as nn
from typing import List, Dict, Tuple, Union
import numpy as np



class KVEmbeddingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, dummy_input, indices, embedding_layer):
        """
        forward 
        ctx:
        dummy_input:
        indices:
        embedding_layer:
        """
        
        unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
        ctx.save_for_backward(indices, unique_indices, inverse_indices)
        ctx.embedding_layer = embedding_layer

        unique_embeddings = []
        for idx in unique_indices:
            key = idx.item()
            embedding_vector = embedding_layer._fetch_vector(key)
            unique_embeddings.append(embedding_vector)

        unique_embeddings = torch.stack(unique_embeddings)

        embeddings = unique_embeddings[inverse_indices] + (dummy_input.sum() * 0.0)

        return embeddings
    
    @staticmethod
    def backward(ctx, grad_output):
        indices, unique_indices, inverse_indices = ctx.saved_tensors
        embedding_layer = ctx.embedding_layer

        # print("Grad Output:", grad_output)
        # print("Grad Output Shape:", grad_output.shape)

        embedding_layer._accumulate_gradients(unique_indices, inverse_indices, grad_output)

        return embedding_layer._grad_dummy, None, None


class KVEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, mean: float, std: float, **kwargs):
        super().__init__()
        self.embedding = {}
        self.embedding_dim = embedding_dim
        self.init_mean = mean
        self.init_std = std
        self._grad_dummy = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self._grad_accumulator = {}
        self.learning_rate = 0.001
        
        self._momentum_state: Dict[int, Tuple[torch.Tensor, torch.Tensor, int]] = {}
        # Adam hyperparameters
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.999)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        self.weight_decay = kwargs.get('weight_decay', 0.0)


    def _init_vector(self,) -> torch.Tensor:
        return torch.nn.init.normal_(torch.empty(self.embedding_dim), mean= self.init_mean, std=self.init_std)
    

    def _fetch_vector(self, key: int) -> torch.Tensor:
        if key not in self.embedding.keys():
            self.embedding[key] = self._init_vector()
        return self.embedding[key]
    
    def _debug_print(self, key: int) -> None:
        if self.embedding.__contains__(key):
            print(f'{key} exists in embedding', self.embedding[key].tolist())
        else:
            print(f'{key} does not exist in embedding')


    def _accumulate_gradients(self, unique_indices, inverse_indices, grad_output):
        flat_grad_output = grad_output.view(-1, self.embedding_dim)
        flat_inverse_indices = inverse_indices.view(-1)

        for i, idx in enumerate(unique_indices):
            key = idx.item()

            mask = (flat_inverse_indices == i)
            if mask.any:
                grad_for_key = flat_grad_output[mask].mean(dim=0)
                # self.embedding[key] -= self.learning_rate * grad_for_key
                # if key not in self._grad_accumulator:
                #     self._grad_accumulator[key] = torch.zeros_like(grad_for_key)
                # self._grad_accumulator[key] += grad_for_key

                # Apply weight decay if specified
                if self.weight_decay > 0:
                    grad_for_key = grad_for_key + self.weight_decay * self.embedding[key]
                

                if key not in self._momentum_state:
                    first_momentum = torch.zeros_like(self.embedding[key])
                    second_momentum = torch.zeros_like(self.embedding[key])
                    time_step = 0
                    self._momentum_state[key] = (first_momentum, second_momentum, time_step)
                # Retrieve momentum state
                first_momentum, second_momentum, time_step = self._momentum_state[key]
                
                # Update time step
                time_step += 1

                # Update biased first moment estimate
                first_momentum = self.beta1 * first_momentum + (1 - self.beta1) * grad_for_key
                
                # Update biased second raw moment estimate
                second_momentum = self.beta2 * second_momentum + (1 - self.beta2) * (grad_for_key ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = first_momentum / (1 - self.beta1 ** time_step)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = second_momentum / (1 - self.beta2 ** time_step)

                self.embedding[key] -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)

                self._momentum_state[key] = (first_momentum, second_momentum,time_step)

                

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # assume indices is a 2D tensor but could be 1D tensor or 3D tensor
        # bz, feat_num = indices.shape 

        # unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
        # # print(unique_indices, inverse_indices)
        # unique_embeddings = []
        # for idx in unique_indices:
        #     key = idx.item()
        #     embedding_vector = self._fetch_vector(key)
        #     unique_embeddings.append(embedding_vector)

        # unique_embeddings = torch.stack(unique_embeddings)

        # embeddings = unique_embeddings[inverse_indices]

        # return embeddings
        return KVEmbeddingFunction.apply(self._grad_dummy, indices, self)