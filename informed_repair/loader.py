from typing import Optional
import torch.utils.data
from heuristic import EditHeuristic

class Loader: # TODO: need more descriptive name
    def __init__(self, 
                 dataset: Optional[torch.utils.data.Dataset] = None, 
                 dataloader: Optional[torch.utils.data.DataLoader] = None, 
                 model: Optional[torch.nn.Module] = None, 
                 batch_size: Optional[int] = None):
        self.dataset: Optional[torch.utils.data.Dataset] = dataset
        self.dataloader: Optional[torch.utils.data.DataLoader] = dataloader
        self.model: Optional[torch.nn.Module] = model
        self.batch_size: Optional[int] = batch_size
        self.heuristic: Optional[Heuristic] = None