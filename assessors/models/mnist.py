from typing import *

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .torch.model import TorchModel, TrainSettings


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


class MNISTDefault(TorchModel):
    module: LightningModule

    def __init__(self):
        self.module = MNISTDefaultModule()
        self.train_settings = TrainSettings()
        super().__init__()

    def name(self) -> str:
        return "mnist_default"
