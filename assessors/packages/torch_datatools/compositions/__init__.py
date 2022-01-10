"""
Compositions are either Torch Datasets that take as input other Torch Datasets,
or functions that return multiple Torch Datasets (and similarly take in Torch Datasets).
"""

from .enumerated_dict import EnumeratedDictDataset
from .skip import SkipDataset
from .take import TakeDataset
from .transform import TransformDataset
from .reference import ReferenceDataset
from .enumerate import EnumerateDataset

from .split_by_ratio import split_by_ratio
from .split_by_count import split_by_count
from .unzip import unzip
