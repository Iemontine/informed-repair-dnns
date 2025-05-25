import models
from loader import Loader
import torch
from torch import nn
import torchvision

from heuristics import FromLayerHeuristic, SetHeuristic

ROOT = "./data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    mlp_test()

def mlp_test():
    model = models.mlp()

    edit_heuristic = FromLayerHeuristic(
        start_layer=-1,
        lb=-0.1,
        ub=0.1,
    )

    set_heuristic = SetHeuristic(
        filename=f"{ROOT}/moon_misclassifications.pt",
        device=DEVICE,
        dtype=torch.float32,
    )

    loader = Loader(
        model=models.mlp(),
        edit_heuristic=edit_heuristic,
        set_heuristic=set_heuristic,
    )


if __name__ == "__main__":
    main()
