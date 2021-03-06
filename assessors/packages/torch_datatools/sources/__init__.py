"""
Sources are Torch Datasets that take as input things that are not Torch Datasets yet.
"""

from .pandas import DataFrameDataset
from .sequence import SequenceDataset
from .range import RangeDataset
