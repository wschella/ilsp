from typing import *
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms


from assessors.core import ModelDefinition, TrainedModel, Restore, Dataset
from assessors.datasets.torch import TorchDataset
from assessors.packages import torch_datatools as tools


class MNISTDefaultNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(28 * 28, 128)
        self.output = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):  # type: ignore
        x = torch.flatten(x, 1)  # flatten all dims except batch
        x = F.relu(self.hidden(x))
        x = self.softmax(self.output(x))
        # TODO: Check if it works with nllloss / cross-entropy
        # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        return x


class MNISTDefault(ModelDefinition):
    epochs = 10

    def name(self) -> str:
        return "mnist_default"

    def train(self, dataset: Dataset, validation: Dataset, restore: Restore) -> TrainedModel:
        if not torch.cuda.is_available():
            logging.warn('Using CPU, this will be slow')
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_ds: td.Dataset = cast(TorchDataset, dataset).ds
        # train_ds = tools.compositions.TransformInputDataset(train_ds, lambda x: x / 255.0)
        # train_ds = tools.compositions.ToDeviceDataset(train_ds, device="cuda:0")
        val_ds: td.Dataset = cast(TorchDataset, validation).ds
        # val_ds = tools.compositions.TransformInputDataset(val_ds, lambda x: x / 255.0)

        # TODO: set pin_memory
        train_loader = td.DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
        val_loader = td.DataLoader(val_ds, batch_size=32, pin_memory=True)

        model = MNISTDefaultNetwork().cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(self.epochs):

            for _batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                val_loss = 0.0
                val_acc = 0.0
                for _batch_idx, (data, target) in enumerate(val_loader):
                    data, target = data.cuda(), target.cuda()
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    val_acc += (output.argmax(dim=1) == target).sum().item()

                val_loss /= len(val_loader)
                val_acc /= len(val_loader)
                print(
                    f"Epoch {epoch}: loss={loss.item():.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")  # type: ignore

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
        torch.save(self.model.state_dict(), path)

    def loss(self, y_true, y_pred) -> float:
        raise NotImplementedError()

    def score(self, y_true, y_pred) -> float:
        raise NotImplementedError()

    def predict(self, entry, **kwargs) -> Any:
        raise NotImplementedError()

    def predict_all(self, dataset, **kwargs) -> Sequence[Any]:
        raise NotImplementedError()
