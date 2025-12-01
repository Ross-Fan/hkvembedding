import torch 
import torch.nn as nn
import numpy as np 


embedding_table = {
    1: torch.tensor([1, 2, 3], dtype=torch.float32),
    2: torch.tensor([2, 5, 6], dtype=torch.float32),
    3: torch.tensor([3, 8, 9], dtype=torch.float32),
    4: torch.tensor([4, 2, 3], dtype=torch.float32),
    5: torch.tensor([5, 5, 6], dtype=torch.float32),
    6: torch.tensor([6, 8, 9], dtype=torch.float32),
    7: torch.tensor([7, 2, 3], dtype=torch.float32),
    8: torch.tensor([8, 5, 6], dtype=torch.float32),
    9: torch.tensor([9, 8, 9], dtype=torch.float32),
}

tensor_2d = torch.tensor([[1, 2, 3],
                          [4, 5, 3],
                          [7, 8, 3]], dtype=torch.int64)

print("2D Tensor:")
print(tensor_2d)
unique_indices, inverse_indices = torch.unique(tensor_2d, return_inverse=True)
print(unique_indices, inverse_indices)

unique_embeddings = []
for idx in unique_indices:
    key = idx.item()
    embedding_vector = embedding_table[key]
    unique_embeddings.append(embedding_vector)

unique_embeddings = torch.stack(unique_embeddings)

embeddings = unique_embeddings[inverse_indices]

print(embeddings)
print(embeddings.shape)
# unique_tensor = torch.unique(tensor_2d)
# print("Unique values:", unique_tensor)

tensor_3d = torch.randint(0, 10, (5, 5, 5))
unique_tensor = torch.unique(tensor_3d)
print("Stensor_3d:", tensor_3d)
print("Shape of tensor_3d:", tensor_3d.shape)
print("Unique values:", unique_tensor) 

c = dict()
c[1]=2 


def adds(a):
    a[2] = 2

adds(c)
print(c)