from enum import Enum, auto
import torch
from torchvision import transforms as T


class Tag(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()

    def __str__(self) -> str:
        return self.name.lower()


RESIZE_CROP = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])


def normalise(tensor: torch.Tensor) -> torch.Tensor:
    """Min-max normalise a tensor for visualisation."""
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
