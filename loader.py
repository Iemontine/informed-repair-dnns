from typing import Optional
import torch.utils.data
from heuristics import EditHeuristic, SetHeuristic

class Loader: # TODO: need more descriptive name
    def __init__(self, 
                dataset: Optional[torch.utils.data.Dataset] = None, 
                model: Optional[torch.nn.Module] = None, 
                batch_size: int = 32,
                edit_heuristic: Optional[EditHeuristic] = None,
                set_heuristic: Optional[SetHeuristic] = None,
                ):
        self.dataset: Optional[torch.utils.data.Dataset] = dataset
        self.model: Optional[torch.nn.Module] = model
        self.batch_size: int = batch_size
        self.heuristic: Optional[EditHeuristic] = edit_heuristic,
        self.editset: Optional[SetHeuristic] = set_heuristic