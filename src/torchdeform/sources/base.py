from abc import abstractmethod, ABC
from torch import nn

from ..core import Displacement


class SourceModel(nn.Module, ABC):
    """
    Generic source displacement model.

    Conventions
    -----------
    - All distances in meters
    - Returns ENU Displacement object in meters
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> Displacement:
        pass