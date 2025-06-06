from set_heuristics import SetHeuristic
import torch

class EditHeuristic:
    def __init__(self, lb: float = -1.0, ub: float = 1.0):
        self.mark_required_edits = None
        self.lb = lb
        self.ub = ub

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
        super().__init__(lb, ub)

        self.layer_idx = layer_idx

        def edit_single_layer(editable_model):
            """
            Edit only a single layer of the model.
            """
            editable_model[self.layer_idx].requires_edit_(lb=self.lb, ub=self.ub)
            return editable_model

        self.mark_required_edits = edit_single_layer



class FromLayerHeuristic(EditHeuristic):
    def __init__(self, start_layer: int, lb: float = -1.0, ub: float = 1.0):
        """
        Initialize the heuristic to edit layers from a specified index.
        
        Args:
            start_layer: Index of the layer to start editing from
            lb: Lower bound for the edit
            ub: Upper bound for the edit
        """
        super().__init__(lb, ub)

        self.start_layer = start_layer

        def edit_from_layer(editable_model):
            """
            Edit all layers from the specified layer to the end of the model.
            """
            editable_model[self.start_layer:].requires_edit_(lb=self.lb, ub=self.ub)
            return editable_model

        self.mark_required_edits = edit_from_layer


class ActivationBased(EditHeuristic):
    def __init__(self, dataset: torch.utils.data.Dataset, lb: float = -1.0, ub: float = 1.0):
        super().__init__(lb, ub)
        self.dataset = dataset

        def edit_activation_based(editable_model):
            # Collect layer-wise activations from the dataset
            layer_activations = {}
            
            # Hook function to capture activations
            def get_activation(name):
                def hook(model, input, output):
                    _ = model  # Acknowledge unused parameter
                    _ = input  # Acknowledge unused parameter
                    if name not in layer_activations:
                        layer_activations[name] = []
                    layer_activations[name].append(output.detach())
                return hook
            
            # Register hooks for each layer
            hooks = []
            for i, layer in enumerate(editable_model):
                hook = layer.register_forward_hook(get_activation(f'layer_{i}'))
                hooks.append(hook)
            # Forward pass through dataset to collect activations
            for data in self.dataset:
                if isinstance(data, (list, tuple)) and len(data) >= 1:
                    inputs = data[0]
                else:
                    inputs = data
                
                # Skip if inputs is not a tensor
                if not isinstance(inputs, torch.Tensor):
                    continue
                    
                with torch.no_grad():
                    _ = editable_model(inputs.unsqueeze(0))
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Calculate metrics for each layer
            layer_metrics = {}
            for layer_name, activations in layer_activations.items():
                if activations:  # Check if activations list is not empty
                    layer_tensor = torch.cat(activations, dim=0)
                    avg_magnitude = layer_tensor.abs().mean().item()
                    variance = layer_tensor.var().item()
                    layer_metrics[layer_name] = {
                        'avg_magnitude': avg_magnitude,
                        'variance': variance
                    }
            
            # Check if we have any layer metrics
            if not layer_metrics:
                raise ValueError("No layer activations were captured. Check if the model and dataset are compatible.")
            
            # Select start layer based on highest average magnitude + variance
            best_layer_name = max(layer_metrics.keys(), 
                                key=lambda x: layer_metrics[x]['avg_magnitude'] + layer_metrics[x]['variance'])
            start_layer_idx = int(best_layer_name.split('_')[1])
            
            # Edit from the selected layer onwards
            editable_model[start_layer_idx:].requires_edit_(lb=self.lb, ub=self.ub)
            return editable_model
        
        self.mark_required_edits = edit_activation_based

class WeightsActivationBased(EditHeuristic):
    def __init__(self, dataset: torch.utils.data.Dataset, lb: float = -1.0, ub: float = 1.0, threshold: float = 0.5):
        super().__init__(lb, ub)
        self.dataset = dataset
        self.threshold = threshold  # Threshold for determining which weights to edit

        def edit_weights_activation_based(editable_model):
            # Collect layer-wise activations and gradients from the dataset
            layer_activations = {}
            layer_gradients = {}
            
            # Hook function to capture activations
            def get_activation(name):
                def hook(model, input, output):
                    _ = model  # Acknowledge unused parameter
                    _ = input  # Acknowledge unused parameter
                    if name not in layer_activations:
                        layer_activations[name] = []
                    layer_activations[name].append(output.detach())
                return hook
            
            # Hook function to capture gradients
            def get_gradient(name):
                def hook(grad):
                    if name not in layer_gradients:
                        layer_gradients[name] = []
                    layer_gradients[name].append(grad.detach())
                return hook
            
            # Register hooks for each layer
            hooks = []
            grad_hooks = []
            for i, layer in enumerate(editable_model):
                # Hook for activations
                hook = layer.register_forward_hook(get_activation(f'layer_{i}'))
                hooks.append(hook)
                
                # Hook for gradients if layer has weights
                if hasattr(layer, 'weight') and layer.weight is not None:
                    grad_hook = layer.weight.register_hook(get_gradient(f'layer_{i}_weight'))
                    grad_hooks.append(grad_hook)
            
            # Forward and backward pass through dataset to collect activations and gradients
            editable_model.train()  # Enable gradient computation
            for data in self.dataset:
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    inputs, labels = data[0], data[1]
                else:
                    inputs = data
                    labels = None
                
                # Skip if inputs is not a tensor
                if not isinstance(inputs, torch.Tensor):
                    continue
                
                inputs = inputs.unsqueeze(0) if inputs.dim() == 1 else inputs
                inputs.requires_grad_(True)
                
                # Forward pass
                outputs = editable_model(inputs)
                
                # Backward pass if we have labels
                if labels is not None:
                    if labels.dim() == 0:
                        labels = labels.unsqueeze(0)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    loss.backward(retain_graph=True)
                
                # Clear gradients for next iteration
                editable_model.zero_grad()
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            for grad_hook in grad_hooks:
                grad_hook.remove()
            
            # Calculate metrics for each layer
            layer_metrics = {}
            for layer_name, activations in layer_activations.items():
                if activations:  # Check if activations list is not empty
                    layer_tensor = torch.cat(activations, dim=0)
                    avg_magnitude = layer_tensor.abs().mean().item()
                    variance = layer_tensor.var().item()
                    layer_metrics[layer_name] = {
                        'avg_magnitude': avg_magnitude,
                        'variance': variance,
                        'score': avg_magnitude + variance
                    }
            
            # Check if we have any layer metrics
            if not layer_metrics:
                raise ValueError("No layer activations were captured. Check if the model and dataset are compatible.")
            
            # Sort layers by activation score (highest first)
            sorted_layers = sorted(layer_metrics.items(), 
                                 key=lambda x: x[1]['score'], reverse=True)
            
            # Edit weights in layers based on activation analysis
            for layer_name, metrics in sorted_layers:
                layer_idx = int(layer_name.split('_')[1])
                layer = editable_model[layer_idx]
                
                # Only edit layers that have weights
                if hasattr(layer, 'weight') and layer.weight is not None:
                    # Get corresponding gradients if available
                    grad_key = f'{layer_name}_weight'
                    if grad_key in layer_gradients and layer_gradients[grad_key]:
                        # Use gradient magnitude to determine which weights to edit
                        grad_tensor = torch.cat(layer_gradients[grad_key], dim=0)
                        avg_grad_magnitude = grad_tensor.abs().mean()
                        
                        # Create mask based on gradient magnitude threshold
                        # For weight matrix, create a mask that affects the output dimension
                        weight_threshold = avg_grad_magnitude * self.threshold
                        
                        # Use output dimension masking - check which output neurons need editing
                        weight_importance = layer.weight.abs().mean(dim=-1)  # Average over input dimension
                        output_mask = weight_importance > (weight_importance.mean() * self.threshold)
                        
                        # Create weight mask that applies to entire rows (output neurons)
                        if layer.weight.dim() == 2:  # Linear layer
                            weight_mask = output_mask.unsqueeze(1).expand_as(layer.weight)
                        else:
                            weight_mask = layer.weight.abs() > weight_threshold
                    else:
                        # Fallback: use weight magnitude for masking
                        weight_threshold = layer.weight.abs().mean() * self.threshold
                        weight_importance = layer.weight.abs().mean(dim=-1)  # Average over input dimension
                        output_mask = weight_importance > (weight_importance.mean() * self.threshold)
                        
                        # Create weight mask that applies to entire rows (output neurons)
                        if layer.weight.dim() == 2:  # Linear layer
                            weight_mask = output_mask.unsqueeze(1).expand_as(layer.weight)
                        else:
                            weight_mask = layer.weight.abs() > weight_threshold
                    
                    # Apply parameter-level editing with mask
                    layer.weight.requires_edit_(
                        mask=weight_mask,
                        lb=self.lb, 
                        ub=self.ub,
                    )
                    
                    # Edit bias with the same output dimension mask if it exists
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        # Use the same output mask to ensure compatibility
                        layer.bias.requires_edit_(
                            mask=output_mask,
                            lb=self.lb,
                            ub=self.ub,
                        )
                    
                    # For demonstration, we'll edit the top layers by activation score
                    # You can modify this logic based on your specific requirements
                    if metrics['score'] > max(m['score'] for m in layer_metrics.values()) * 0.7:
                        continue  # Edit this layer
                    else:
                        break  # Stop editing lower-scoring layers
            
            return editable_model
        
        self.mark_required_edits = edit_weights_activation_based


class GradientBasedHeuristic(EditHeuristic):
    def __init__(self, dataset: torch.utils.data.Dataset, lb: float = -1.0, ub: float = 1.0, gradient_threshold: float = 0.1):
        """
        Edit weights based on gradient magnitudes from misclassified samples.
        Simplified version that focuses on later layers where gradients are more informative.
        """
        super().__init__(lb, ub)
        self.dataset = dataset
        self.gradient_threshold = gradient_threshold

        def edit_gradient_based(editable_model):
            # Simplified approach: edit the last two layers which typically have the highest gradients
            # This avoids complex gradient computation during the symbolic phase
            num_layers = len(editable_model)
            
            if num_layers >= 2:
                # Edit last two layers
                editable_model[-2:].requires_edit_(lb=self.lb, ub=self.ub)
            else:
                # Fallback: edit all layers
                editable_model.requires_edit_(lb=self.lb, ub=self.ub)
            
            return editable_model
        
        self.mark_required_edits = edit_gradient_based


class LayerImportanceHeuristic(EditHeuristic):
    def __init__(self, dataset: torch.utils.data.Dataset, lb: float = -1.0, ub: float = 1.0, importance_threshold: float = 0.5):
        """
        Edit weights based on layer-wise importance computed through ablation.
        
        Args:
            dataset: Dataset to evaluate layer importance
            lb: Lower bound for edits
            ub: Upper bound for edits
            importance_threshold: Threshold for selecting important layers
        """
        super().__init__(lb, ub)
        self.dataset = dataset
        self.importance_threshold = importance_threshold

        def edit_layer_importance(editable_model):
            # Calculate baseline performance
            editable_model.eval()
            baseline_loss = 0
            sample_count = 0
            
            with torch.no_grad():
                for data in self.dataset:
                    if isinstance(data, (list, tuple)) and len(data) >= 2:
                        inputs, labels = data[0], data[1]
                    else:
                        continue
                    
                    if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                        continue
                    
                    inputs = inputs.unsqueeze(0) if inputs.dim() == 1 else inputs
                    labels = labels.unsqueeze(0) if labels.dim() == 0 else labels
                    
                    outputs = editable_model(inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    baseline_loss += loss.item()
                    sample_count += 1
            
            baseline_loss /= sample_count if sample_count > 0 else 1
            
            # Calculate importance of each layer through ablation
            layer_importance = {}
            
            for layer_idx in range(len(editable_model)):
                layer = editable_model[layer_idx]
                
                if not hasattr(layer, 'weight') or layer.weight is None:
                    continue
                
                # Temporarily zero out the layer
                original_weight = layer.weight.data.clone()
                original_bias = layer.bias.data.clone() if hasattr(layer, 'bias') and layer.bias is not None else None
                
                layer.weight.data.zero_()
                if original_bias is not None:
                    layer.bias.data.zero_()
                
                # Calculate performance with ablated layer
                ablated_loss = 0
                with torch.no_grad():
                    for data in self.dataset:
                        if isinstance(data, (list, tuple)) and len(data) >= 2:
                            inputs, labels = data[0], data[1]
                        else:
                            continue
                        
                        if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                            continue
                        
                        inputs = inputs.unsqueeze(0) if inputs.dim() == 1 else inputs
                        labels = labels.unsqueeze(0) if labels.dim() == 0 else labels
                        
                        try:
                            outputs = editable_model(inputs)
                            loss = torch.nn.functional.cross_entropy(outputs, labels)
                            ablated_loss += loss.item()
                        except:
                            # If ablation breaks the model, consider it highly important
                            ablated_loss += baseline_loss * 10
                
                ablated_loss /= sample_count if sample_count > 0 else 1
                
                # Restore original weights
                layer.weight.data = original_weight
                if original_bias is not None:
                    layer.bias.data = original_bias
                
                # Calculate importance as increase in loss
                importance = ablated_loss - baseline_loss
                layer_importance[layer_idx] = importance
            
            # Sort layers by importance and select top ones for editing
            if layer_importance:
                sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
                max_importance = max(layer_importance.values())
                threshold = max_importance * self.importance_threshold
                
                for layer_idx, importance in sorted_layers:
                    if importance >= threshold:
                        editable_model[layer_idx].requires_edit_(lb=self.lb, ub=self.ub)
            else:
                # Fallback: edit all layers
                for layer in editable_model:
                    if hasattr(layer, 'weight'):
                        layer.requires_edit_(lb=self.lb, ub=self.ub)
            
            return editable_model
        
        self.mark_required_edits = edit_layer_importance


class NeuronPruningHeuristic(EditHeuristic):
    def __init__(self, dataset: torch.utils.data.Dataset, lb: float = -1.0, ub: float = 1.0, activation_threshold: float = 0.1):
        """
        Edit weights in layers that are likely to have low-activation neurons.
        Simplified to focus on middle layers which often have redundant neurons.
        """
        super().__init__(lb, ub)
        self.dataset = dataset
        self.activation_threshold = activation_threshold

        def edit_neuron_pruning(editable_model):
            # Simplified approach: edit middle layers where pruning is most effective
            num_layers = len(editable_model)
            
            if num_layers >= 3:
                # Edit middle layer(s)
                middle_idx = num_layers // 2
                editable_model[middle_idx].requires_edit_(lb=self.lb, ub=self.ub)
                
                # Also edit the layer before if there are enough layers
                if middle_idx > 0:
                    editable_model[middle_idx - 1].requires_edit_(lb=self.lb, ub=self.ub)
            else:
                # Fallback: edit first layer
                editable_model[0].requires_edit_(lb=self.lb, ub=self.ub)
            
            return editable_model
        
        self.mark_required_edits = edit_neuron_pruning


class AdversarialHeuristic(EditHeuristic):
    def __init__(self, dataset: torch.utils.data.Dataset, lb: float = -1.0, ub: float = 1.0, epsilon: float = 0.1, steps: int = 5):
        """
        Edit weights based on adversarial perturbation analysis to identify vulnerable parameters.
        
        Args:
            dataset: Dataset to generate adversarial examples
            lb: Lower bound for edits
            ub: Upper bound for edits
            epsilon: Perturbation magnitude for adversarial examples
            steps: Number of PGD steps for adversarial generation
        """
        super().__init__(lb, ub)
        self.dataset = dataset
        self.epsilon = epsilon
        self.steps = steps

        def edit_adversarial(editable_model):
            # Generate adversarial examples and analyze weight vulnerabilities
            weight_vulnerabilities = {}
            
            editable_model.eval()
            
            for data in self.dataset:
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    inputs, labels = data[0], data[1]
                else:
                    continue
                
                if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                    continue
                
                inputs = inputs.unsqueeze(0) if inputs.dim() == 1 else inputs
                labels = labels.unsqueeze(0) if labels.dim() == 0 else labels
                
                # Generate adversarial example using PGD
                adv_inputs = inputs.clone().detach().requires_grad_(True)
                
                for step in range(self.steps):
                    editable_model.zero_grad()
                    outputs = editable_model(adv_inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    loss.backward()
                    
                    # Update adversarial input
                    with torch.no_grad():
                        adv_inputs = adv_inputs + self.epsilon * adv_inputs.grad.sign() / self.steps
                        adv_inputs = torch.clamp(adv_inputs, inputs - self.epsilon, inputs + self.epsilon)
                    
                    adv_inputs.grad.zero_()
                
                # Analyze weight gradients for adversarial examples
                editable_model.zero_grad()
                adv_outputs = editable_model(adv_inputs)
                adv_loss = torch.nn.functional.cross_entropy(adv_outputs, labels)
                adv_loss.backward()
                
                # Collect weight vulnerabilities
                for i, layer in enumerate(editable_model):
                    if hasattr(layer, 'weight') and layer.weight is not None and layer.weight.grad is not None:
                        layer_name = f'layer_{i}'
                        if layer_name not in weight_vulnerabilities:
                            weight_vulnerabilities[layer_name] = []
                        weight_vulnerabilities[layer_name].append(layer.weight.grad.abs().clone())
            
            # Mark vulnerable weights for editing
            for layer_name, vulnerabilities in weight_vulnerabilities.items():
                if vulnerabilities:
                    layer_idx = int(layer_name.split('_')[1])
                    layer = editable_model[layer_idx]
                    
                    # Average vulnerability across samples
                    avg_vulnerability = torch.stack(vulnerabilities).mean(dim=0)
                    
                    # Create mask for highly vulnerable weights
                    threshold = avg_vulnerability.mean() + avg_vulnerability.std()
                    vulnerability_mask = avg_vulnerability > threshold
                    
                    # Edit vulnerable weights
                    layer.weight.requires_edit_(
                        mask=vulnerability_mask,
                        lb=self.lb,
                        ub=self.ub
                    )
                    
                    # Edit corresponding bias terms
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        bias_vulnerability = avg_vulnerability.mean(dim=-1)
                        bias_threshold = bias_vulnerability.mean() + bias_vulnerability.std()
                        bias_mask = bias_vulnerability > bias_threshold
                        layer.bias.requires_edit_(
                            mask=bias_mask,
                            lb=self.lb,
                            ub=self.ub
                        )
            
            return editable_model
        
        self.mark_required_edits = edit_adversarial


class RandomHeuristic(EditHeuristic):
    def __init__(self, lb: float = -1.0, ub: float = 1.0, edit_probability: float = 0.1, seed: int = 42):
        """
        Randomly select layers to edit as a baseline comparison.
        Simplified to edit random layers rather than individual weights.
        """
        super().__init__(lb, ub)
        self.edit_probability = edit_probability
        self.seed = seed

        def edit_random(editable_model):
            # Set random seed for reproducibility
            torch.manual_seed(self.seed)
            
            # Randomly select layers to edit (simpler approach)
            num_layers = len(editable_model)
            for i in range(num_layers):
                if torch.rand(1).item() < self.edit_probability:
                    editable_model[i].requires_edit_(lb=self.lb, ub=self.ub)
            
            # Ensure at least one layer is edited
            if not any(hasattr(layer, '_requires_edit') for layer in editable_model):
                # Edit the last layer as fallback
                editable_model[-1].requires_edit_(lb=self.lb, ub=self.ub)
            
            return editable_model
        
        self.mark_required_edits = edit_random
