import models
from loader import Loader
import torch
from torch import nn
import torchvision

from heuristics import ActivationBased, SetHeuristic

ROOT = "./data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    loader = Loader(
        dataset=torchvision.datasets.MNIST(
            root=ROOT,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda t: t.flatten(-3,-1)),
            ])
        ),
        model = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        ).to(device=DEVICE, dtype=torch.float32),
        edit_heuristic=ActivationBased,
        set_heuristic=SetHeuristic, # TODO: implement a different set heuristic
    )

    # network = models.vit_b_32(pretrained=True, eval=True)
    # print(network)


if __name__ == "__main__":
    main()
