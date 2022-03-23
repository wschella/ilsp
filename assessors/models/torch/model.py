# https://pytorch.org/tutorials/beginner/saving_loading_models.html

from typing import *
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from assessors.core import Model, Dataset, Restore
from assessors.datasets.torch import TorchDataset


@dataclass
class TrainSettings:
    epochs: int = 10
    batch_size: int = 256
    val_batch_size: int = 256
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True
    num_workers: int = 12


class TorchModel(Model, ABC):
    """
    An abstract base model for PyTorch models implementing the assessors.core.Model interface.
    """

    module: LightningModule
    train_settings: TrainSettings

    # --------------------------------------------------------------------------
    # Unimplemented abstract methods from Model
    # --------------------------------------------------------------------------
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def predict(self, entry, **kwargs) -> Any:
        pass

    @abstractmethod
    def predict_all(self, dataset, **kwargs) -> Sequence[Any]:
        pass

    @abstractmethod
    def score(self, y_true, y_pred) -> float:
        pass

    @abstractmethod
    def loss(self, y_true, y_pred) -> float:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @abstractmethod
    def try_restore_from(self, path: Path) -> Optional[Tuple[()]]:
        pass

    # --------------------------------------------------------------------------
    # Implemented abstract methods from Model
    # --------------------------------------------------------------------------

    def train(self, dataset: Dataset, validation: Dataset, restore: Restore, **kwargs):
        settings = self.train_settings
        loader_kwargs: Dict[str, Any] = {
            "pin_memory": settings.pin_memory,
            "num_workers": settings.num_workers,
            "drop_last": settings.drop_last,
        }

        train_ds: td.Dataset = cast(TorchDataset, dataset).ds
        train_loader = td.DataLoader(
            train_ds,
            batch_size=settings.batch_size,
            shuffle=True,
            **loader_kwargs,
        )

        val_ds: td.Dataset = cast(TorchDataset, validation).ds
        val_loader = td.DataLoader(
            val_ds,
            batch_size=settings.batch_size,
            **loader_kwargs)

        model = self.module
        trainer = Trainer(
            max_epochs=settings.epochs,
            gpus=1,
            logger=TensorBoardLogger('./artifacts/lightning_logs'),
            auto_select_gpus=True,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
        )
        trainer.fit(model, train_loader, val_loader)
