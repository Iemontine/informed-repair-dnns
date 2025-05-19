from typing import Optional
import torch.utils.data
from heuristic import EditHeuristic

class Loader: # TODO: need more descriptive name
    def __init__(self, 
                 dataset: Optional[torch.utils.data.Dataset] = None, 
                 model: Optional[torch.nn.Module] = None, 
                 batch_size: int = 32):
        self.dataset: Optional[torch.utils.data.Dataset] = dataset
        self.model: Optional[torch.nn.Module] = model
        self.batch_size: int = batch_size
        self.heuristic: Optional[EditHeuristic] = None
        self.editset: Optional[torch.utils.data.Dataset] = None