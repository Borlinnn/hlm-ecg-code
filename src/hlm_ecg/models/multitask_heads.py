"""Small auxiliary heads for HLM-ECG ablations."""

from torch import nn


class SubclassAuxiliaryHead(nn.Module):
    """Linear subclass head attached to the shared A3 feature."""

    def __init__(self, *, in_features: int, num_subclasses: int) -> None:
        super().__init__()
        if int(num_subclasses) <= 0:
            raise ValueError("num_subclasses must be positive")
        self.num_subclasses = int(num_subclasses)
        self.fc = nn.Linear(int(in_features), self.num_subclasses)

    def forward(self, features):
        return self.fc(features)
