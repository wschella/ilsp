from typing import *

import assessors.models as models
import assessors.core as core
from assessors.core import ModelDefinition, DatasetDescription


class SystemHub():
    systems: Dict[str, Dict[str, Type[ModelDefinition]]] = {
        'mnist': {
            # "default": models.MNISTDefault,
        },
        'cifar10': {
            # "default": models.CIFAR10Default,
            # "wide": models.CIFAR10Wide,
        },
        'segment': {
            # "default": models.SegmentDefault,
        },
    }

    @staticmethod
    def get(dataset: str, model: str) -> Type[ModelDefinition]:
        return SystemHub.systems[dataset][model]

    @staticmethod
    def exists(dataset: str, model: str) -> bool:
        return dataset in SystemHub.systems and model in SystemHub.systems[dataset]

    @staticmethod
    def options_for(dataset) -> List[str]:
        return list(SystemHub.systems[dataset].keys())


class AssessorHub():
    assessors: Dict[str, Dict[str, Type[ModelDefinition]]] = {
        'mnist': {
            # "default": models.MNISTAssessorDefault,
        },
        'cifar10': {
            # "default": models.CIFAR10AssessorDefault,
            # "wide": models.CIFAR10AssessorWide,
        },
        'segment': {
            # "default": models.SegmentAssessorDefault
        },
    }

    @staticmethod
    def get(dataset: str, model: str) -> Type[ModelDefinition]:
        return AssessorHub.assessors[dataset][model]

    @staticmethod
    def exists(dataset: str, model: str) -> bool:
        return dataset in AssessorHub.assessors and model in AssessorHub.assessors[dataset]

    @staticmethod
    def options_for(dataset) -> List[str]:
        return list(AssessorHub.assessors[dataset].keys())


class DatasetHub():
    datasets: Dict[str, DatasetDescription] = {
        'mnist': core.TorchVisionMNIST,
        'cifar10': core.TorchVisionCIFAR10,
        'segment': core.OpenMLSegment,
    }

    @staticmethod
    def get(name: str) -> DatasetDescription:
        return DatasetHub.datasets[name]

    @staticmethod
    def options() -> List[str]:
        return list(DatasetHub.datasets.keys())

    @staticmethod
    def exists(name: str) -> bool:
        return name in DatasetHub.datasets
