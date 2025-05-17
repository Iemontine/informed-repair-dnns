import sytorch as st
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