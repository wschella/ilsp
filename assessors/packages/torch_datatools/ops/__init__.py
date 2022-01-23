"""
Ops are either Torch Datasets that take as input other Torch Datasets,
or functions that return multiple Torch Datasets (and similarly take in Torch Datasets).
I.e. they are operations under which the set of Torch Datasets is closed.
"""

from .enumerated_dict import EnumeratedDictDataset
from .skip import SkipDataset
from .take import TakeDataset
from .transform import TransformDataset
from .transform_input import TransformInputDataset
from .transform_target import TransformTargetDataset
from .reference import ReferenceDataset
from .enumerate import EnumerateDataset
from .interleave import InterleaveDataset
from .to_device import ToDeviceDataset

from .split_by_ratio import split_by_ratio
from .split_by_count import split_by_count
from .unzip import unzip
