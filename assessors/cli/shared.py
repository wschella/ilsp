from typing import *
from abc import ABC, abstractmethod

import assessors.models as models
from assessors.core import ModelDefinition, TFDatasetDescription, DatasetDescription, CSVDatasetDescription


def get_model_def(dataset, model_name) -> type[ModelDefinition]:
    model: type[ModelDefinition] = {  # type: ignore
        'mnist': models.MNISTDefault,
        'cifar10': models.CIFAR10Default,
        'segment': models.SegmentDefault,
    }[dataset]
    return model


def get_assessor_def(dataset, model_name) -> type[ModelDefinition]:
    model: type[ModelDefinition] = {  # type: ignore
        'mnist': {
            "default": models.MNISTAssessorDefault,
        },
        'cifar10': {
            "default": models.CIFAR10AssessorDefault,
        },
        'segment': {
            "default": models.SegmentAssessorDefault,
        }
    }[dataset][model_name]
    return model


def get_dataset_description(dataset) -> DatasetDescription:
    return {
        'mnist': TFDatasetDescription('mnist'),
        'cifar10': TFDatasetDescription('cifar10'),
        'segment': CSVDatasetDescription('segment_brodley.csv'),
    }[dataset]


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
