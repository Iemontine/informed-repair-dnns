from typing import Optional
import torch.utils.data
from edit_heuristics import EditHeuristic, SetHeuristic
from models import convert_model_to_editable


class Loader: # TODO: need more descriptive name
    def __init__(self,
                model: Optional[torch.nn.Module] = None,
                batch_size: int = 32,
                edit_heuristic: Optional[EditHeuristic] = None,
                set_heuristic: Optional[SetHeuristic] = None,
                dataset: Optional[torch.utils.data.TensorDataset] = None,
                ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model: Optional[torch.nn.Module] = model
        self.editable_model: torch.nn.Module = convert_model_to_editable(model) if model is not None else None
        
        self.batch_size: int = batch_size
        self.edit_heuristic: Optional[EditHeuristic] = edit_heuristic
        self.set_heuristic: Optional[SetHeuristic] = set_heuristic
        self.dataset: Optional[torch.utils.data.TensorDataset] = dataset


    def edit_and_test_model(self) -> torch.Tensor:
        editable_model = self.edit_heuristic.edit(
            editable_model=self.editable_model,
            set_heuristic=self.set_heuristic
        )

        # Convert the editable model to a standard PyTorch model
        self.model.eval()

        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)
        correct = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.model(inputs)
                loss = criterion(output, targets)
                total_loss += loss.item() * inputs.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()

        accuracy = correct / len(dataloader.dataset)
        avg_loss = total_loss / len(dataloader.dataset)

        print(f"Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")

        return accuracy, avg_loss
