from __future__ import annotations
from typing import *
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .dataset import Dataset


class Model(ABC):
    @abstractmethod
    def name(self) -> str:
        """
        The name of the model.
        """
        pass

    @abstractmethod
    def train(self, dataset: Dataset, validation: Dataset, restore: Restore, **kwargs):
        pass

    def __call__(self, entry, **kwargs):
        return self.predict(entry, **kwargs)

    @abstractmethod
    def predict(self, entry, **kwargs) -> Any:
        pass

    @abstractmethod
    def predict_all(self, dataset, **kwargs) -> Sequence[Any]:
        pass

    @abstractmethod
    def score(self, y_true, y_pred) -> float:
        """
        Calculate the score for the given output and target.
        E.g. 0 or 1 for a classification task.
        TODO: This should be moved somewhere else.
        """
        pass

    @abstractmethod
    def loss(self, y_true, y_pred) -> float:
        """
        Calculate the loss for the given output and target, with the loss function
        used to train this model.
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save all learnable parameters and all learning state to disk.
        """
        pass

    @abstractmethod
    def try_restore_from(self, path: Path) -> Optional[Tuple[()]]:
        """
        Try to restore saved parameters from disk.

        Returns
        -------
        None or empty tuple
            If None, then no parameters were restored.
        """
        pass

    def restore_from(self, path: Path) -> Tuple[()]:
        """
        Restore saved parameters from disk or error if that fails.

        Returns
        -------
        empty tuple
            Returns the empty tuple if parameters and state were restored successfully.
        """
        result = self.try_restore_from(path)
        if result is None:
            raise ValueError(f'Model not found at {path}')
        return ()


# TODO: Add assessor model definition, wich has a specific init or set syst_id and syst_features input size

@dataclass
class Restore:
    path: Path
    option: Union[Literal["full"], Literal["checkpoint"], Literal["off"]] = "off"
    Options = Union[Literal["full"], Literal["checkpoint"], Literal["off"]]

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
                pass
                # print(f"Restoring {name} from {self.path}")
            else:
                print(f"Restoring {name} from checkpoint {self.path}")
