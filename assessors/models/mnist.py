from typing import *
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from assessors.core import ModelDefinition, TrainedModel, Restore, Dataset
from assessors.datasets.torch import TorchDataset


class MNISTDefaultModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(28 * 28, 128)
        self.output = torch.nn.Linear(128, 10)

    def forward(self, x):  # type: ignore
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.output(x)
        # x = torch.relu(x)
        # TODO: Check if it works with nllloss / cross-entropy
        # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.htm
        x = torch.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_nb):  # type: ignore
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_nb):  # type: ignore
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):  # type: ignore
        return torch.optim.Adam(self.parameters(), lr=0.02)


class MNISTDefault(ModelDefinition):
    epochs = 10

    def name(self) -> str:
        return "mnist_default"

    def train(self, dataset: Dataset, validation: Dataset, restore: Restore) -> TrainedModel:
        train_ds: td.Dataset = cast(TorchDataset, dataset).ds
        train_loader = td.DataLoader(
            train_ds, batch_size=256, shuffle=True, drop_last=True,
            pin_memory=True, num_workers=12)

        val_ds: td.Dataset = cast(TorchDataset, validation).ds
        val_loader = td.DataLoader(
            val_ds, batch_size=512,
            pin_memory=True, num_workers=12)

        model = MNISTDefaultModule()
        trainer = Trainer(
            max_epochs=self.epochs,
            gpus=1,
            logger=TensorBoardLogger('./artifacts/lightning_logs'),
            callbacks=[TQDMProgressBar(refresh_rate=20)],
        )
        trainer.fit(model, train_loader, val_loader)

        return TorchTrainedModel(model)

    def try_restore_from(self, path: Path) -> Optional[TrainedModel]:
        raise NotImplementedError()

    def score(self, y_true, y_pred) -> float:
        raise NotImplementedError()


class TorchTrainedModel(TrainedModel):
    model: nn.Module

    def __init__(self, model) -> None:
        self.model = model
        super().__init__()

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path / "model.pt")

    def loss(self, y_true, y_pred) -> float:
        raise NotImplementedError()

    def score(self, y_true, y_pred) -> float:
        raise NotImplementedError()

    def predict(self, entry, **kwargs) -> Any:
        raise NotImplementedError()

    def predict_all(self, dataset, **kwargs) -> Sequence[Any]:
        raise NotImplementedError()
