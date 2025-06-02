from typing import Optional
from pathlib import Path

from copy import deepcopy
import sytorch as st
import torch
import torch.nn as nn
import torchvision
# from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights

# # Largest patch size (should be easier to repair and evaluate)
# def vit_b_32(pretrained: bool = True, eval: bool = True):
#     weights = ViT_B_32_Weights.DEFAULT if pretrained else None
#     network = torchvision.models.vit_b_32(weights=weights).train(mode = not eval)
#     # network = st.nn.to_editable(network)
#     return network

# # Use if vit_b_32 is too coarse
# def vit_b_16(pretrained: bool = True, eval: bool = True):
#     weights = ViT_B_16_Weights.DEFAULT if pretrained else None
#     network = torchvision.models.vit_b_16(weights=weights).train(mode = not eval)
#     # network = st.nn.to_editable(network)
#     return network

def mlp(path: Path = None,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> nn.Sequential:
    model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2 classes
    ).to(device=device, dtype=dtype)
    if path is not None:
        model.load_state_dict(
            torch.load(path, weights_only=True, map_location=device)
        )

    return model

def convert_model_to_editable(model: nn.Module) -> nn.Module:
    editable_model = deepcopy(model)                    # Copy model to not modify original
    editable_model = st.nn.to_editable(editable_model)  # Convert to editable symbolic model
    editable_model.solver.verbose_(False)               # Disable solver verbose output
    return editable_model