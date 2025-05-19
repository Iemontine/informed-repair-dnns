from typing import Optional
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


class SetHeuristic:
    def __init__(self):
        self.editset = None

    def load_edit_set(self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        shape: tuple = (3, 32, 32),
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