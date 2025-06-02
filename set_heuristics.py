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
        
        # Normalize inputs to unit vectors for proper cosine similarity
        inputs_normalized = torch.nn.functional.normalize(inputs_flat, p=2, dim=1)
        similarities = torch.mm(inputs_normalized, inputs_normalized.t())  # Cosine similarity
        
        # Remove self-similarities (diagonal) to get a better sense of the similarity distribution
        mask = ~torch.eye(similarities.size(0), dtype=torch.bool, device=self.device)
        similarities_no_diag = similarities[mask]
        
        # Scale threshold based on the actual similarity distribution
        mean_sim = similarities_no_diag.mean()
        std_sim = similarities_no_diag.std()
        
        # Use a scaled threshold: mean + (threshold * std)
        # This makes the threshold adaptive to the actual similarity range in your data
        scaled_threshold = mean_sim + (self.similarity_threshold * std_sim)
        
        # Clamp to reasonable bounds
        scaled_threshold = torch.clamp(scaled_threshold, -1.0, 1.0)

        # Create adjacency matrix based on scaled similarity threshold
        adjacency = (similarities > scaled_threshold).float()

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


class MisclassifiedSetHeuristic(SetHeuristic):
    """
    Create a set heuristic from all misclassified samples.
    This heuristic can either:
    1. Load pre-computed misclassified samples from a file (more efficient)
    2. Evaluate the model on the dataset to find misclassifications (fallback)
    """
    def __init__(self, 
        model: torch.nn.Module = None,
        dataset: torch.utils.data.Dataset = None,
        filename: Optional[str] = None,
        device: torch.device = torch.device('cpu'), 
        dtype: torch.dtype = torch.float32, 
        max_samples: Optional[int] = None
    ):
        """
        Initialize MisclassifiedSetHeuristic.
        
        Args:
            model: The model to evaluate for misclassifications (used if filename is None)
            dataset: Dataset to evaluate (used if filename is None)
            filename: Path to pre-computed misclassifications file (more efficient if available)
            device: Device to run computations on
            dtype: Data type for tensors
            max_samples: Maximum number of misclassified samples to collect (-1 for all)
        """
        self.model = model
        self.dataset = dataset
        self.max_samples = max_samples
        
        if filename is not None: # Use pre-computed misclassifications from file (more efficient)
            misclassified_features, misclassified_labels = self._load_from_file(filename)
        elif model is not None and dataset is not None: # Compute misclassifications by evaluating model on dataset
            misclassified_features, misclassified_labels = self._find_misclassified_samples()
        else:
            raise ValueError("Either filename or (model + dataset) must be provided")
        
        # Initialize parent with the misclassified samples
        super().__init__(
            features=misclassified_features, 
            labels=misclassified_labels, 
            device=device, 
            dtype=dtype
        )
    
    def _load_from_file(self, filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load misclassified samples from a pre-computed file."""
        data = torch.load(filename)
        if isinstance(data, dict) and 'features' in data and 'labels' in data:
            features = data['features']
            labels = data['labels']
            
            # Apply max_samples limit if specified
            if self.max_samples is not None and len(features) > self.max_samples:
                features = features[:self.max_samples]
                labels = labels[:self.max_samples]
                
            return features, labels
        else:
            raise ValueError(f"File {filename} does not contain expected 'features' and 'labels' keys")
    
    
    def _find_misclassified_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find all misclassified samples in the dataset."""
        misclassified_features = []
        misclassified_labels = []
        samples_collected = 0
        
        self.model.eval()
        with torch.no_grad():
            for data in self.dataset:
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    inputs, labels = data[0], data[1]
                else:
                    # Handle case where dataset only contains features
                    inputs = data
                    continue  # Skip if no labels available
                
                if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                    continue
                
                # Add batch dimension if needed
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                # Get model predictions
                outputs = self.model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                
                # Find misclassified samples
                misclassified_mask = predicted != labels
                
                if misclassified_mask.any():
                    misclassified_inputs = inputs[misclassified_mask]
                    misclassified_targets = labels[misclassified_mask]
                    
                    for i in range(misclassified_inputs.size(0)):
                        if self.max_samples is None or samples_collected < self.max_samples:
                            misclassified_features.append(misclassified_inputs[i])
                            misclassified_labels.append(misclassified_targets[i])
                            samples_collected += 1
                        else:
                            break
                    
                    if self.max_samples is not None and samples_collected >= self.max_samples:
                        break
        
        if not misclassified_features:
            raise ValueError("No misclassified samples found in the dataset")
        
        return torch.stack(misclassified_features), torch.stack(misclassified_labels)


class ByClassSetHeuristic(SetHeuristic):
    """
    Create a set heuristic containing all samples from a specific class.
    This heuristic selects all samples that belong to a target class,
    regardless of whether they were correctly classified or not.
    """
    def __init__(self, 
        dataset: torch.utils.data.Dataset,
        target_class: int,
        device: torch.device = torch.device('cpu'), 
        dtype: torch.dtype = torch.float32, 
        max_samples: Optional[int] = None
    ):
        """
        Initialize ByClassSetHeuristic.
        
        Args:
            dataset: Dataset to sample from
            target_class: The class index to collect samples for
            device: Device to run computations on
            dtype: Data type for tensors
            max_samples: Maximum number of samples to collect (-1 for all)
        """
        self.dataset = dataset
        self.target_class = target_class
        self.max_samples = max_samples
        
        # Find samples of the target class
        class_features, class_labels = self._find_class_samples()
        
        # Initialize parent with the class samples
        super().__init__(
            features=class_features, 
            labels=class_labels, 
            device=device, 
            dtype=dtype
        )
    
    def _find_class_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find all samples belonging to the target class."""
        class_features = []
        class_labels = []
        samples_collected = 0
        
        for data in self.dataset:
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                inputs, labels = data[0], data[1]
            else:
                # Handle case where dataset only contains features
                inputs = data
                continue  # Skip if no labels available
            
            if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                continue
            
            # Add batch dimension if needed
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            # Find samples of the target class
            target_class_mask = labels == self.target_class
            
            if target_class_mask.any():
                target_inputs = inputs[target_class_mask]
                target_labels = labels[target_class_mask]
                
                for i in range(target_inputs.size(0)):
                    if self.max_samples is None or self.max_samples == -1 or samples_collected < self.max_samples:
                        class_features.append(target_inputs[i])
                        class_labels.append(target_labels[i])
                        samples_collected += 1
                    else:
                        break
                
                if self.max_samples is not None and self.max_samples != -1 and samples_collected >= self.max_samples:
                    break
        
        if not class_features:
            raise ValueError(f"No samples found for class {self.target_class} in the dataset")
        
        return torch.stack(class_features), torch.stack(class_labels)


class FullDatasetHeuristic(SetHeuristic):
    """
    A set heuristic that includes the entire dataset.
    This heuristic doesn't filter or select specific samples - it uses all available data.
    """
    def __init__(self, 
        dataset: torch.utils.data.Dataset,
        filename: Optional[str] = None,
        features: torch.Tensor = None, 
        labels: torch.Tensor = None, 
        device: torch.device = torch.device('cpu'), 
        dtype: torch.dtype = torch.float32, 
        max_samples: Optional[int] = None
    ):
        self.dataset = dataset
        self.max_samples = max_samples
        
        if dataset is not None:
            # Extract all features and labels from the dataset
            all_features = []
            all_labels = []
            
            for i, data in enumerate(dataset):
                if self.max_samples is not None and i >= self.max_samples:
                    break
                    
                if isinstance(data, (tuple, list)) and len(data) >= 2:
                    inputs, labels = data[0], data[1]
                else:
                    # Handle case where dataset only contains features
                    inputs = data
                    continue  # Skip if no labels available
                
                if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                    continue
                
                # Add batch dimension if needed
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                all_features.append(inputs)
                all_labels.append(labels)
            
            if not all_features:
                raise ValueError("No valid samples found in the dataset")
            
            # Convert to tensors
            features = torch.cat(all_features, dim=0).to(dtype)
            labels = torch.cat(all_labels, dim=0)
            
            # Call parent constructor with extracted features and labels
            super().__init__(None, features, labels, device, dtype, None)
        else:
            # Fall back to parent constructor for file-based or direct tensor input
            super().__init__(filename, features, labels, device, dtype, None)