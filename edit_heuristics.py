from set_heuristics import SetHeuristic
import torch

class EditHeuristic:
    def __init__(self):
        self.mark_required_edits = None


    def edit(
            self,
            editable_model: torch.nn.Module,    # Editable model to optimize
            set_heuristic: SetHeuristic,             
            lb: float = -0.2,                   # Lower bound for parameter changes
            ub: float = 0.2,                    # Upper bound for parameter changes
    ) -> torch.nn.Module:
        editable_model = self.mark_required_edits(editable_model)

        inputs, labels = set_heuristic.get_inputs_and_labels()

        # Perform symbolic forward pass
        symbolic_outputs = editable_model(inputs).data

        # Assert that the edited model must correctly classify all inputs
        output_constraints = symbolic_outputs.argmax(-1) == labels
        output_constraints.assert_()

        # Define the optimization objective
        objective = editable_model.param_delta().norm_ub('linf+l1n')
        
        # Optimize the model with the given objective
        if editable_model.optimize(minimize=objective):
            return editable_model


class SingleLayerHeuristic(EditHeuristic):
    def __init__(self, layer_idx: int, lb: float = -1.0, ub: float = 1.0):
        """
        Initialize the heuristic to edit a single layer.
        
        Args:
            layer_idx: Index of the layer to edit
            lb: Lower bound for the edit
            ub: Upper bound for the edit
        """
        def edit_single_layer(editable_model):
            """
            Edit only a single layer of the model.
            """
            editable_model[self.layer_idx].requires_edit_(lb=self.lb, ub=self.ub)
            return editable_model

        self.mark_required_edits = edit_single_layer
        self.layer_idx = layer_idx
        self.lb = lb
        self.ub = ub


class FromLayerHeuristic(EditHeuristic):
    def __init__(self, start_layer: int, lb: float = -1.0, ub: float = 1.0):
        """
        Initialize the heuristic to edit layers from a specified index.
        
        Args:
            start_layer: Index of the layer to start editing from
            lb: Lower bound for the edit
            ub: Upper bound for the edit
        """

        def edit_from_layer(editable_model):
            """
            Edit all layers from the specified layer to the end of the model.
            """
            editable_model[self.start_layer:].requires_edit_(lb=self.lb, ub=self.ub)
            return editable_model

        super().__init__()
        self.mark_required_edits = edit_from_layer
        self.start_layer = start_layer
        self.lb = lb
        self.ub = ub


class ActivationBased(EditHeuristic):
    def __init__(self):
        super().__init__()


    # TODO: completely ai-generated & untested
    
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
        """Edit the layer with the highest activation magnitude."""
        layer_idx = self.select_layer_by_activation(model, dataset, device, method='highest')
        return self.edit_single_layer(model, layer_idx, lb, ub)
    
    def edit_from_highest_activation(self, model, dataset, lb, ub, device=torch.device('cpu')):
        """Edit all layers starting from the one with highest activation magnitude."""
        start_layer = self.select_layer_by_activation(model, dataset, device, method='highest')
        return self.edit_from_layer(model, start_layer, lb, ub)