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
    
class SimilaritySetHeuristic(SetHeuristic):
    def __init__(self, 
        filename: Optional[str] = None,
        features: torch.Tensor = None, 
        labels: torch.Tensor = None, 
        device: torch.device = torch.device('cpu'), 
        dtype: torch.dtype = torch.float32, 
        size: Optional[int] = None,
        similarity_threshold: float = 0.5,
    ):
        super().__init__(filename, features, labels, device, dtype, size)
        self.similarity_threshold = similarity_threshold

    def get_inputs_and_labels(self):
        inputs, labels = super().get_inputs_and_labels()

        # By scanning each of the inputs in the edit set, find the subset that includes the most similar inputs
        # Compute pairwise cosine similarities
        inputs_flat = inputs.view(inputs.size(0), -1)  # Flatten inputs for similarity computation
        similarities = torch.mm(inputs_flat, inputs_flat.t())  # Dot product
        norms = torch.norm(inputs_flat, dim=1, keepdim=True)
        similarities = similarities / (norms * norms.t())  # Cosine similarity

        # Create adjacency matrix based on similarity threshold
        adjacency = (similarities > self.similarity_threshold).float()

        # Find connected components (groups of similar inputs)
        visited = torch.zeros(inputs.size(0), dtype=torch.bool, device=self.device)
        largest_component = []
        largest_size = 0

        for i in range(inputs.size(0)):
            if not visited[i]:
                # BFS to find connected component
                component = []
                queue = [i]
                while queue:
                    node = queue.pop(0)
                    if not visited[node]:
                        visited[node] = True
                        component.append(node)
                        # Add neighbors
                        neighbors = torch.where(adjacency[node] == 1)[0]
                        for neighbor in neighbors:
                            if not visited[neighbor]:
                                queue.append(neighbor.item())
                
                # Update largest component if this one is bigger
                if len(component) > largest_size:
                    largest_component = component
                    largest_size = len(component)

        # Filter inputs and labels to only include the largest similar group
        if largest_component:
            indices = torch.tensor(largest_component, device=self.device)
            inputs = inputs[indices]
            labels = labels[indices]

        return inputs, labels