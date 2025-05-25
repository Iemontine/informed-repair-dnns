import sytorch as st
import torch
import torch.nn as nn
import torchvision
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights

# Largest patch size (should be easier to repair and evaluate)
def vit_b_32(pretrained: bool = True, eval: bool = True):
    weights = ViT_B_32_Weights.DEFAULT if pretrained else None
    network = torchvision.models.vit_b_32(weights=weights).train(mode = not eval)
    network = st.nn.to_editable(network)
    return network

# Use if vit_b_32 is too coarse
def vit_b_16(pretrained: bool = True, eval: bool = True):
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    network = torchvision.models.vit_b_16(weights=weights).train(mode = not eval)
    network = st.nn.to_editable(network)
    return network

def mlp():
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)  # 2 classes
            )
            
        def forward(self, x):
            return self.layers(x)


    loaded_model = MLP()
    loaded_model.load_state_dict(torch.load('data/moon_classifier_mlp.pth'))
    loaded_model.eval()
    return loaded_model