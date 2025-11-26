import torch
import numpy as np
from typing import List, Union, Tuple

def hash_ids(ids: Union[List[int], np.ndarray, torch.Tensor], 
             hash_size: int = None) -> np.ndarray:
    """
    对ID进行哈希处理
    
    Args:
        ids: 输入ID
        hash_size: 哈希表大小，如果为None则不进行哈希
        
    Returns:
        哈希后的ID
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.cpu().numpy()
    elif isinstance(ids, list):
        ids = np.array(ids)
    
    if hash_size is not None:
        ids = ids % hash_size
    
    return ids.astype(np.uint64)
