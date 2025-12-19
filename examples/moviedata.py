import torch 
import pandas as pd
from torch.utils.data import Dataset

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
