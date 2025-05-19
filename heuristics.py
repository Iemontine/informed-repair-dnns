from typing import Optional, Tuple
import torch


class EditHeuristic:
    def edit_single_layer(self, model, layer_idx, lb, ub):
        """
        Edit only a single layer of the model.
        """
        model[layer_idx].requires_edit_(lb=lb, ub=ub)
        return model

    def edit_from_layer(self, model, start_layer, lb, ub):
        """
        Edit all layers from the specified layer to the end of the model.
        """
        model[start_layer:].requires_edit_(lb=lb, ub=ub)
        return model

    def edit_with_mask(self, model, layer_idx, lb, ub, param_name="weight", mask=None, condition=None):
        """
        Edit part of the parameters in a layer, specified by a mask.
        Args:
            mask: Boolean mask specifying which parameters to edit.
            param_name: Parameter to edit ("weight" or "bias")
            condition: Function that creates a mask (e.g., lambda x: x > 0)

            mask and condition should be mutually exclusive.
            If both are provided, mask will be used.
        """

        # TODO: add support for parameter selection based on some output of model, 
        # e.g. activation, gradient/sensitivity based
        # currently, supports simple masks only

        param = getattr(model[layer_idx], param_name)
        
        if mask is None and condition is not None:
            mask = condition(param)
        
        param.requires_edit_(mask=mask, lb=lb, ub=ub)
        return model
    

class ActivationBased(EditHeuristic):
    def __init__(self):
        super().__init__()
    
    def _collect_activations(self, model, dataset, device=torch.device('cpu')):
        """
        Collect activations from all layers in the model for a dataset.
        
        Args:
            model: PyTorch model (assumed to be Sequential)
            dataset: Dataset to collect activations on
            device: Device to run computations on
            
        Returns:
            List of average activation magnitudes for each layer
        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Prepare to store activations
        activations = [0] * len(model)
        counts = [0] * len(model)
        
        # Set up hooks
        handles = []
        
        def get_hook(layer_idx):
            def hook(module, input, output):
                # Average the absolute values of the activations
                activations[layer_idx] += torch.mean(torch.abs(output)).item()
                counts[layer_idx] += 1
            return hook
        
        # Register hooks for all layers
        for i in range(len(model)):
            if hasattr(model[i], 'register_forward_hook'):
                handle = model[i].register_forward_hook(get_hook(i))
                handles.append(handle)
        
        # Run forward pass
        model.eval()
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                _ = model(inputs)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Calculate average activation magnitude for each layer
        for i in range(len(activations)):
            if counts[i] > 0:
                activations[i] /= counts[i]
        
        return activations
    
    def select_layer_by_activation(self, model, dataset, device=torch.device('cpu'), method='highest'):
        """
        Select a layer based on activation statistics.
        
        Args:
            model: PyTorch model (assumed to be Sequential)
            dataset: Dataset to collect activations on
            device: Device to run computations on
            method: Selection method ('highest' for highest average activation magnitude)
            
        Returns:
            Index of the selected layer
        """
        activations = self._collect_activations(model, dataset, device)
        
        if method == 'highest':
            return activations.index(max(activations))
        elif method == 'lowest':
            return activations.index(min([a for a in activations if a > 0]))
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def edit_highest_activation_layer(self, model, dataset, lb, ub, device=torch.device('cpu')):
        """
        Edit the layer with the highest activation magnitude.
        
        Args:
            model: PyTorch model (assumed to be Sequential)
            dataset: Dataset to collect activations on
            lb: Lower bound for edits
            ub: Upper bound for edits
            device: Device to run computations on
        
        Returns:
            Edited model
        """
        layer_idx = self.select_layer_by_activation(model, dataset, device, method='highest')
        return self.edit_single_layer(model, layer_idx, lb, ub)
    
    def edit_from_highest_activation(self, model, dataset, lb, ub, device=torch.device('cpu')):
        """
        Edit all layers starting from the one with highest activation magnitude.
        
        Args:
            model: PyTorch model (assumed to be Sequential)
            dataset: Dataset to collect activations on
            lb: Lower bound for edits
            ub: Upper bound for edits
            device: Device to run computations on
        
        Returns:
            Edited model
        """
        start_layer = self.select_layer_by_activation(model, dataset, device, method='highest')
        return self.edit_from_layer(model, start_layer, lb, ub)


class SetHeuristic:
    def __init__(self):
        self.editset = None

    def load_edit_set(self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        shape: Tuple[int, ...] = (3, 32, 32),   # TODO: this is a bad default
        size: Optional[int] = None,
    ) -> torch.utils.data.TensorDataset:
        images = images.to(device=device, dtype=dtype).reshape(-1, *shape) / 255.
        labels = labels.to(device=device, dtype=torch.int64)

        if size is not None:
            images = images[:size]
            labels = labels[:size]

        return torch.utils.data.TensorDataset(
            images,
            labels
        )