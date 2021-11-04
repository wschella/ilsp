from __future__ import annotations
from dataclasses import dataclass
from typing import *
from abc import ABC, abstractmethod
from pathlib import Path


class ModelDefinition(ABC):

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def train(self, dataset, validation, restore: Restore) -> TrainedModel:
        pass

    @abstractmethod
    def try_restore_from(self, path: Path) -> Optional[TrainedModel]:
        pass


class TrainedModel(ABC):
    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @abstractmethod
    def evaluate(self, dataset, callback):
        """
        Should log the instance wise evaluation results to file as CSV with folllowing columns:
        y_true, y_pred, prediction, loss, is_correct
        """
        pass

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def predict(self, entry, **kwargs):
        pass

    def __call__(self, entry, **kwargs):
        return self.predict(entry, **kwargs)


@dataclass
class Restore:
    path: Path
    option: Union[Literal["full"], Literal["checkpoint"], Literal["off"]] = "off"

    @staticmethod
    def FROM_SCRATCH(save_path: Path) -> Restore:
        """
        You still need to provide a path because we need a place to save model checkpoints.
        This can not be turned off currently.
        """
        return Restore(save_path, "off")

    @staticmethod
    def FULL(path: Path) -> Restore:
        return Restore(path, "full")

    @staticmethod
    def CHECKPOINT(path: Path) -> Restore:
        return Restore(path, "checkpoint")

    def should_restore_full(self) -> bool:
        return self.option == "full"

    def should_restore_checkpoint(self) -> bool:
        return self.option == "checkpoint"

    def should_restore(self) -> bool:
        return self.option != "off"

    def log(self, name: str) -> None:
        if not self.should_restore():
            print(f'{name} will not be restored due to restore="off"')
        else:
            if self.should_restore_full():
                print(f"Restoring {name} from {self.path}")
            else:
                print(f"Restoring {name} from checkpoint {self.path}")
