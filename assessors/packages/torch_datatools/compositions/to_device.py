from typing import *

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ._shared import T_co

from ..sources import SequenceDataset


class ToDeviceDataset(Dataset[Tensor]):
    """
    A Dataset of Tensors that will all be send to the given device (i.e. indices).
    """
    source: Dataset[Tensor]
    device: str
    at_once: bool

    def __init__(
        self,
        source: Dataset[Tensor],
        device: str = "cuda:0",
        at_once: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        source : Dataset[Tensor]
            The source dataset.
        device : str, default 'cuda:0'
            The device to send the tensors to.
        at_once : bool, default True
            If True, the tensors will be send to the device at once during the
            setup of this instance. This require iterating over the entire dataset
            and holding the entire dataset in device memory.
        """
        length = len(source)  # type: ignore
        self.at_once = at_once
        self.device = device

        if self.at_once:
            self.source = SequenceDataset([
                source[i].to(device=self.device) for i in range(length)
            ])
        else:
            self.source = source

        super().__init__()

    def __getitem__(self, index) -> Tensor:
        if self.at_once:
            return self.source[index]
        else:
            tensor = self.source[index]
            return tensor.to(device=self.device)

    def __len__(self) -> int:
        return len(self.source)  # type: ignore
