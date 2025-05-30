from typing import Optional, Tuple
import torch
from torch import Tensor

class SetHeuristic:
    def __init__(self, 
        filename: Optional[str] = None,
        features: torch.Tensor = None, 
        labels: torch.Tensor = None, 
        device: torch.device = torch.device('cpu'), 
        dtype: torch.dtype = torch.float32, 
        size: Optional[int] = None
    ):
        # These also set the value of self.features and self.labels
        self.editset: Optional[torch.utils.data.TensorDataset] = None
        self.device = device
        if filename is not None:
            self.editset = self._load_edit_set_from_file(
                filename, dtype, size
            )
        elif features is not None and labels is not None:
            self.editset = self._load_edit_set_from_features_labels(
                features, labels, dtype, size
            )
        else:
            raise ValueError("Either filename or both features and labels must be provided.")
        

    def get_inputs_and_labels(self) -> Tuple[Tensor, Tensor]:
        """
        Extract inputs and labels from the set heuristic.
        
        Args:
            set_heuristic: SetHeuristic instance containing editset
            
        Returns:
            Tuple of (inputs, labels)
        """
        # This doesn't work for some reason
        # inputs = set_heuristic.features[0]
        # labels = set_heuristic.labels[0]
        inputs = []
        labels = []
        for _input, _label in self.editset:
            inputs.append(_input)
            labels.append(_label)
        inputs = torch.stack(inputs).to(self.device)
        labels = torch.stack(labels).to(self.device)
        return inputs, labels


    def _load_edit_set_from_file(self,
        filename: str,
        dtype: torch.dtype = torch.float32,
        size: Optional[int] = None,
    ) -> torch.utils.data.TensorDataset:
        """
        Load a dataset from a file and return it as a TensorDataset.
        """
        data = torch.load(filename)
        features = data['features']
        labels = data['labels']

        return self._load_edit_set_from_features_labels(features, labels, dtype, size)


    def _load_edit_set_from_features_labels(self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        dtype: torch.dtype,
        size: Optional[int],
    ) -> torch.utils.data.TensorDataset:
        self.features = features.to(device=self.device, dtype=dtype) # .reshape(-1, *shape) / 255. # <-- used for colors where shape was shape: Tuple[int, ...] = (3, 32, 32),
        self.labels = labels.to(device=self.device, dtype=torch.int64)

        if size is not None:
            self.features = features[:size]
            self.labels = labels[:size]

        return torch.utils.data.TensorDataset(
            self.features,
            self.labels
        )
    

class SubsetSetHeuristic(SetHeuristic):
    def __init__(self, 
        filename: Optional[str] = None,
        features: torch.Tensor = None, 
        labels: torch.Tensor = None, 
        device: torch.device = torch.device('cpu'), 
        dtype: torch.dtype = torch.float32, 
        size: Optional[int] = None,
        indices: Optional[torch.Tensor] = None,
    ):
        super().__init__(filename, features, labels, device, dtype, size)
        self.indices = indices


    def get_inputs_and_labels(self) -> Tuple[Tensor, Tensor]:
        inputs, labels = super().get_inputs_and_labels()

        if self.indices is not None:
            inputs = inputs[self.indices]
            labels = labels[self.indices]
        else:
            raise ValueError("Indices must be provided for SubsetSetHeuristic.")

        return inputs, labels