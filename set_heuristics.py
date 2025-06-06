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


class ConfidenceBasedHeuristic(SetHeuristic):
    """
    A set heuristic that focuses on samples where the model has low confidence.
    These are typically samples near decision boundaries or difficult cases.
    """
    def __init__(self, 
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        device: torch.device = torch.device('cpu'), 
        dtype: torch.dtype = torch.float32, 
        confidence_threshold: float = 0.8,
        max_samples: Optional[int] = None
    ):
        """
        Initialize ConfidenceBasedHeuristic.
        
        Args:
            model: The model to evaluate confidence
            dataset: Dataset to sample from
            device: Device to run computations on
            dtype: Data type for tensors
            confidence_threshold: Maximum confidence for sample selection (lower = less confident)
            max_samples: Maximum number of samples to collect
        """
        self.model = model
        self.dataset = dataset
        self.confidence_threshold = confidence_threshold
        self.max_samples = max_samples
        
        # Find low-confidence samples
        low_confidence_features, low_confidence_labels = self._find_low_confidence_samples()
        
        # Initialize parent with the low-confidence samples
        super().__init__(
            features=low_confidence_features, 
            labels=low_confidence_labels, 
            device=device, 
            dtype=dtype
        )
    
    def _find_low_confidence_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find samples where the model has low confidence."""
        low_confidence_features = []
        low_confidence_labels = []
        samples_collected = 0
        
        self.model.eval()
        with torch.no_grad():
            for data in self.dataset:
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    inputs, labels = data[0], data[1]
                else:
                    inputs = data
                    continue  # Skip if no labels available
                
                if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                    continue
                
                # Add batch dimension if needed
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                # Get model predictions and confidence
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                max_confidence = torch.max(probabilities, dim=1)[0]
                
                # Find low-confidence samples
                low_confidence_mask = max_confidence < self.confidence_threshold
                
                if low_confidence_mask.any():
                    low_conf_inputs = inputs[low_confidence_mask]
                    low_conf_labels = labels[low_confidence_mask]
                    
                    for i in range(low_conf_inputs.size(0)):
                        if self.max_samples is None or samples_collected < self.max_samples:
                            low_confidence_features.append(low_conf_inputs[i])
                            low_confidence_labels.append(low_conf_labels[i])
                            samples_collected += 1
                        else:
                            break
                    
                    if self.max_samples is not None and samples_collected >= self.max_samples:
                        break
        
        if not low_confidence_features:
            raise ValueError(f"No low-confidence samples found with threshold {self.confidence_threshold}")
        
        return torch.stack(low_confidence_features), torch.stack(low_confidence_labels)


class BoundarySamplesHeuristic(SetHeuristic):
    """
    A set heuristic that focuses on samples near decision boundaries.
    These samples are often the most informative for model repair.
    """
    def __init__(self, 
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        device: torch.device = torch.device('cpu'), 
        dtype: torch.dtype = torch.float32, 
        boundary_margin: float = 0.1,
        max_samples: Optional[int] = None
    ):
        """
        Initialize BoundarySamplesHeuristic.
        
        Args:
            model: The model to analyze decision boundaries
            dataset: Dataset to sample from
            device: Device to run computations on
            dtype: Data type for tensors
            boundary_margin: Margin from decision boundary to consider (smaller = closer to boundary)
            max_samples: Maximum number of samples to collect
        """
        self.model = model
        self.dataset = dataset
        self.boundary_margin = boundary_margin
        self.max_samples = max_samples
        
        # Find boundary samples
        boundary_features, boundary_labels = self._find_boundary_samples()
        
        # Initialize parent with the boundary samples
        super().__init__(
            features=boundary_features, 
            labels=boundary_labels, 
            device=device, 
            dtype=dtype
        )
    
    def _find_boundary_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find samples close to decision boundaries."""
        boundary_features = []
        boundary_labels = []
        samples_collected = 0
        
        self.model.eval()
        with torch.no_grad():
            for data in self.dataset:
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    inputs, labels = data[0], data[1]
                else:
                    inputs = data
                    continue  # Skip if no labels available
                
                if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                    continue
                
                # Add batch dimension if needed
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                # Get model outputs (logits)
                outputs = self.model(inputs)
                
                # Calculate margin to decision boundary
                if outputs.size(1) == 2:  # Binary classification
                    # For binary case, margin is the absolute difference between logits
                    margin = torch.abs(outputs[:, 0] - outputs[:, 1])
                else:  # Multi-class
                    # For multi-class, use difference between top two predictions
                    sorted_outputs = torch.sort(outputs, dim=1, descending=True)[0]
                    margin = sorted_outputs[:, 0] - sorted_outputs[:, 1]
                
                # Find samples close to boundary (small margin)
                boundary_mask = margin < self.boundary_margin
                
                if boundary_mask.any():
                    boundary_inputs = inputs[boundary_mask]
                    boundary_targets = labels[boundary_mask]
                    
                    for i in range(boundary_inputs.size(0)):
                        if self.max_samples is None or samples_collected < self.max_samples:
                            boundary_features.append(boundary_inputs[i])
                            boundary_labels.append(boundary_targets[i])
                            samples_collected += 1
                        else:
                            break
                    
                    if self.max_samples is not None and samples_collected >= self.max_samples:
                        break
        
        if not boundary_features:
            raise ValueError(f"No boundary samples found with margin {self.boundary_margin}")
        
        return torch.stack(boundary_features), torch.stack(boundary_labels)


class ClusteringBasedHeuristic(SetHeuristic):
    """
    A set heuristic that groups similar samples using clustering and selects 
    representative samples from each cluster.
    """
    def __init__(self, 
        dataset: torch.utils.data.Dataset,
        device: torch.device = torch.device('cpu'), 
        dtype: torch.dtype = torch.float32, 
        n_clusters: int = 5,
        samples_per_cluster: int = 2,
        max_samples: Optional[int] = None
    ):
        """
        Initialize ClusteringBasedHeuristic.
        
        Args:
            dataset: Dataset to cluster
            device: Device to run computations on
            dtype: Data type for tensors
            n_clusters: Number of clusters to create
            samples_per_cluster: Number of samples to select from each cluster
            max_samples: Maximum total number of samples to collect
        """
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.samples_per_cluster = samples_per_cluster
        self.max_samples = max_samples
        
        # Find clustered samples
        clustered_features, clustered_labels = self._find_clustered_samples()
        
        # Initialize parent with the clustered samples
        super().__init__(
            features=clustered_features, 
            labels=clustered_labels, 
            device=device, 
            dtype=dtype
        )
    
    def _find_clustered_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find representative samples from each cluster."""
        # First, collect all features and labels
        all_features = []
        all_labels = []
        
        for data in self.dataset:
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                inputs, labels = data[0], data[1]
            else:
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
            raise ValueError("No valid samples found for clustering")
        
        # Convert to tensors
        features_tensor = torch.cat(all_features, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        
        # Flatten features for clustering
        features_flat = features_tensor.view(features_tensor.size(0), -1)
        
        # Simple k-means clustering
        selected_features = []
        selected_labels = []
        
        if features_flat.size(0) < self.n_clusters:
            # If we have fewer samples than clusters, return all samples
            return features_tensor, labels_tensor
        
        # Initialize cluster centers randomly
        cluster_centers = features_flat[torch.randperm(features_flat.size(0))[:self.n_clusters]]
        
        # Run k-means for a few iterations
        for _ in range(10):  # Simple k-means with fixed iterations
            # Assign samples to nearest cluster
            distances = torch.cdist(features_flat, cluster_centers)
            cluster_assignments = torch.argmin(distances, dim=1)
            
            # Update cluster centers
            new_centers = []
            for k in range(self.n_clusters):
                cluster_mask = cluster_assignments == k
                if cluster_mask.any():
                    new_centers.append(features_flat[cluster_mask].mean(dim=0))
                else:
                    new_centers.append(cluster_centers[k])  # Keep old center if no samples
            cluster_centers = torch.stack(new_centers)
        
        # Select representative samples from each cluster
        samples_collected = 0
        for k in range(self.n_clusters):
            cluster_mask = cluster_assignments == k
            if cluster_mask.any():
                cluster_features = features_tensor[cluster_mask]
                cluster_labels = labels_tensor[cluster_mask]
                
                # Select samples closest to cluster center
                cluster_features_flat = features_flat[cluster_mask]
                distances_to_center = torch.norm(cluster_features_flat - cluster_centers[k], dim=1)
                sorted_indices = torch.argsort(distances_to_center)
                
                # Take the closest samples
                n_samples = min(self.samples_per_cluster, cluster_features.size(0))
                if self.max_samples is not None:
                    n_samples = min(n_samples, self.max_samples - samples_collected)
                
                for i in range(n_samples):
                    idx = sorted_indices[i]
                    selected_features.append(cluster_features[idx])
                    selected_labels.append(cluster_labels[idx])
                    samples_collected += 1
                    
                    if self.max_samples is not None and samples_collected >= self.max_samples:
                        break
            
            if self.max_samples is not None and samples_collected >= self.max_samples:
                break
        
        if not selected_features:
            raise ValueError("No samples selected during clustering")
        
        return torch.stack(selected_features), torch.stack(selected_labels)


class HardNegativeMiningHeuristic(SetHeuristic):
    """
    A set heuristic that focuses on the most difficult examples - those with highest loss.
    """
    def __init__(self, 
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        device: torch.device = torch.device('cpu'), 
        dtype: torch.dtype = torch.float32, 
        max_samples: Optional[int] = None,
        loss_percentile: float = 0.8
    ):
        """
        Initialize HardNegativeMiningHeuristic.
        
        Args:
            model: The model to evaluate sample difficulty
            dataset: Dataset to sample from
            device: Device to run computations on
            dtype: Data type for tensors
            max_samples: Maximum number of samples to collect
            loss_percentile: Percentile threshold for high-loss samples (0.8 = top 20%)
        """
        self.model = model
        self.dataset = dataset
        self.max_samples = max_samples
        self.loss_percentile = loss_percentile
        
        # Find hard negative samples
        hard_features, hard_labels = self._find_hard_samples()
        
        # Initialize parent with the hard samples
        super().__init__(
            features=hard_features, 
            labels=hard_labels, 
            device=device, 
            dtype=dtype
        )
    
    def _find_hard_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find samples with highest loss (most difficult)."""
        sample_losses = []
        all_features = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for data in self.dataset:
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    inputs, labels = data[0], data[1]
                else:
                    inputs = data
                    continue  # Skip if no labels available
                
                if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                    continue
                
                # Add batch dimension if needed
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                # Calculate loss for each sample
                outputs = self.model(inputs)
                losses = torch.nn.functional.cross_entropy(outputs, labels, reduction='none')
                
                for i in range(inputs.size(0)):
                    sample_losses.append(losses[i].item())
                    all_features.append(inputs[i])
                    all_labels.append(labels[i])
        
        if not sample_losses:
            raise ValueError("No valid samples found for hard negative mining")
        
        # Sort by loss (highest first)
        sorted_indices = sorted(range(len(sample_losses)), key=lambda i: sample_losses[i], reverse=True)
        
        # Select top percentile
        threshold_idx = int(len(sorted_indices) * (1 - self.loss_percentile))
        hard_indices = sorted_indices[:max(1, len(sorted_indices) - threshold_idx)]
        
        # Limit to max_samples if specified
        if self.max_samples is not None:
            hard_indices = hard_indices[:self.max_samples]
        
        # Extract hard samples
        hard_features = [all_features[i] for i in hard_indices]
        hard_labels = [all_labels[i] for i in hard_indices]
        
        return torch.stack(hard_features), torch.stack(hard_labels)


class ProgressiveDifficultyHeuristic(SetHeuristic):
    """
    A set heuristic that starts with easier samples and progressively includes harder ones.
    """
    def __init__(self, 
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        device: torch.device = torch.device('cpu'), 
        dtype: torch.dtype = torch.float32, 
        max_samples: Optional[int] = None,
        difficulty_ratio: float = 0.5
    ):
        """
        Initialize ProgressiveDifficultyHeuristic.
        
        Args:
            model: The model to evaluate sample difficulty
            dataset: Dataset to sample from
            device: Device to run computations on
            dtype: Data type for tensors
            max_samples: Maximum number of samples to collect
            difficulty_ratio: Ratio of easy to hard samples (0.5 = half easy, half hard)
        """
        self.model = model
        self.dataset = dataset
        self.max_samples = max_samples
        self.difficulty_ratio = difficulty_ratio
        
        # Find progressive difficulty samples
        progressive_features, progressive_labels = self._find_progressive_samples()
        
        # Initialize parent with the progressive samples
        super().__init__(
            features=progressive_features, 
            labels=progressive_labels, 
            device=device, 
            dtype=dtype
        )
    
    def _find_progressive_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find samples with progressive difficulty ordering."""
        sample_losses = []
        all_features = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for data in self.dataset:
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    inputs, labels = data[0], data[1]
                else:
                    inputs = data
                    continue  # Skip if no labels available
                
                if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                    continue
                
                # Add batch dimension if needed
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                # Calculate loss for each sample
                outputs = self.model(inputs)
                losses = torch.nn.functional.cross_entropy(outputs, labels, reduction='none')
                
                for i in range(inputs.size(0)):
                    sample_losses.append(losses[i].item())
                    all_features.append(inputs[i])
                    all_labels.append(labels[i])
        
        if not sample_losses:
            raise ValueError("No valid samples found for progressive difficulty")
        
        # Sort by loss (lowest first for easy samples)
        sorted_indices = sorted(range(len(sample_losses)), key=lambda i: sample_losses[i])
        
        # Calculate number of easy and hard samples
        total_samples = self.max_samples if self.max_samples is not None else len(sorted_indices)
        n_easy = int(total_samples * self.difficulty_ratio)
        n_hard = total_samples - n_easy
        
        # Select easy samples (low loss) and hard samples (high loss)
        easy_indices = sorted_indices[:n_easy]
        hard_indices = sorted_indices[-n_hard:] if n_hard > 0 else []
        
        # Combine easy and hard samples
        progressive_indices = easy_indices + hard_indices
        
        # Extract progressive samples
        progressive_features = [all_features[i] for i in progressive_indices]
        progressive_labels = [all_labels[i] for i in progressive_indices]
        
        return torch.stack(progressive_features), torch.stack(progressive_labels)