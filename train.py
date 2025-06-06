from loader import Loader
from copy import deepcopy
from itertools import product
import torch


def train_combinations(edit_heuristics_options, set_heuristics_options, original_model, moon_dataset, X, y):
    print(f"Total combinations to test: {len(edit_heuristics_options)} edit × {len(set_heuristics_options)} set = {len(edit_heuristics_options) * len(set_heuristics_options)} combinations")

    # Store results organized by set heuristic
    results_by_set = {}
    for set_name, _ in set_heuristics_options:
            results_by_set[set_name] = []

    print("Testing all combinations of heuristics...")
    for (edit_name, edit_heur), (set_name, set_heur) in product(edit_heuristics_options, set_heuristics_options):
        try:
            test_loader = Loader(
                model=deepcopy(original_model),
                edit_heuristic=edit_heur,
                set_heuristic=set_heur,
                dataset=moon_dataset,
            )
            
            # Edit and test model
            test_loader.edit_and_test_model()
            
            # Get edited model predictions
            with torch.no_grad():
                outputs = test_loader.model(X)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted.numpy() == y.numpy()).mean()
            
            result = {
                'edit_heuristic': edit_name,
                'set_heuristic': set_name,
                'model': deepcopy(test_loader.model),
                'accuracy': accuracy,
                'combination': f"{edit_name} + {set_name}"
            }
            
            results_by_set[set_name].append(result)
            
            print(f"Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error with {edit_name} + {set_name}: {str(e)}")
            print(f"Skipping this combination...")
            
    return results_by_set