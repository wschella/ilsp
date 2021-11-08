from typing import *
from abc import ABC, abstractmethod

import assessors.models as models
from assessors.models import ModelDefinition


class PredictionRecord(TypedDict):
    """
    Will actually be a TF FeaturesDict [1] at runtime.
    But we only care about the field names and their types, the API to access
    them is the same.

    [1] https://www.tensorflow.org/datasets/api_docs/python/tfds/features/FeaturesDict.
    """
    inst_index: Any
    inst_features: Any
    inst_label: Any
    syst_features: Any
    syst_prediction: Any
    syst_pred_loss: Any


def get_model_def(dataset, model_name) -> type[ModelDefinition]:
    model: type[ModelDefinition] = {  # type: ignore
        'mnist': models.MNISTDefault,
        'cifar10': models.CIFAR10Default,
    }[dataset]
    return model


def get_assessor_def(dataset, model_name) -> type[ModelDefinition]:
    model: type[ModelDefinition] = {  # type: ignore
        'mnist': {
            "default": models.MNISTAssessorProbabilistic,
            "prob": models.MNISTAssessorProbabilistic,
        }
        # 'cifar10': models.CIFAR10Default,
    }[dataset][model_name]
    return model


class CommandArguments(ABC):
    """
    An abstract class for the various dataclasses representing command arguments to extend.
    """

    @abstractmethod
    def validate(self):
        pass

    def validated(self):
        self.validate()
        return self

    def validate_option(self, arg_name, options):
        value = getattr(self, arg_name)
        if value not in options:
            raise ValueError(f"Unknown {arg_name} {value}. Options are {options}.")
