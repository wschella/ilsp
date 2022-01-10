from typing import *

from torch.utils.data import Dataset

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
T_contra = TypeVar('T_contra', contravariant=True)


def out_of_bounds(index, ds: Dataset):
    length = len(ds)  # type: ignore
    raise ValueError(f"Index {index} is out of bounds for dataset with length {length}")
