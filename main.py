import models
from loader import Loader
import torch
from torch import nn
import torchvision

from edit_heuristics import FromLayerHeuristic, SingleLayerHeuristic
from set_heuristics import SetHeuristic, SubsetSetHeuristic

ROOT = "./data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    mlp_test()

def mlp_test():
    editable_model = models.mlp(path=f'{ROOT}/moon_classifier_mlp.pth')

    moon_data = torch.load(f"{ROOT}/moon_dataset.pt")
    moon_dataset = torch.utils.data.TensorDataset(moon_data['features'], moon_data['labels'])

    # edit_heuristic = FromLayerHeuristic(
    #     start_layer=-1,
    #     lb=-0.1,
    #     ub=0.1,
    # )
    edit_heuristic = SingleLayerHeuristic(
        layer_idx=-1,
        lb=-0.1,
        ub=0.1,
    )

    set_heuristic = SetHeuristic(
        filename=f"{ROOT}/moon_misclassifications.pt",
        device=DEVICE,
        dtype=torch.float32,
    )

    set_heuristic = SubsetSetHeuristic(set_heuristic, indices=torch.tensor([2]))

    loader = Loader(
        editable_model=editable_model,
        edit_heuristic=edit_heuristic,
        set_heuristic=set_heuristic,
        dataset=moon_dataset,
    )

    loader.edit_and_test_model() 


if __name__ == "__main__":
    main()
