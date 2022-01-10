from typing import *

from torch.utils.data import Dataset

from ._shared import T_co


class InterleaveDataset(Dataset[T_co]):
    """
    A dataset of interleaved datasets.
    Assumes (and enforces) that all datasets have the same length.

    Examples
    --------
    >>> from torch_datatools.compositions import InterleaveDataset
    >>> from torch_datatools.sources import SequenceDataset
    >>> dataset = InterleaveDataset([SequenceDataset([1, 2, 3]), SequenceDataset([4, 5, 6])])
    >>> (dataset[0], dataset[1], dataset[2])
    (1, 4, 2)
    """

    datasets: Sequence[Dataset[T_co]]

    block_length: int
    remainder_block_length: int
    last_idx_in_full_block: int

    def __init__(
        self,
        datasets: Sequence[Dataset[T_co]],
        block_length: int = 1,
    ) -> None:
        assert all(len(d) == len(datasets[0]) for d in datasets)  # type: ignore
        self.datasets = datasets
        self.block_length = block_length

        # Precalculate some values
        length = len(self.datasets[0])  # type: ignore
        n_datasets = len(self.datasets)
        n_full_blocks = (length // self.block_length) * n_datasets
        last_idx_in_full_block = n_full_blocks * self.block_length
        remainder_size = len(self) - last_idx_in_full_block - 1
        remainder_block_length = remainder_size // n_datasets

        self.remainder_block_length = remainder_block_length
        self.last_idx_in_full_block = last_idx_in_full_block

        super().__init__()

    def __getitem__(self, index) -> T_co:
        if index < self.last_idx_in_full_block:
            block_length = self.block_length
        else:
            block_length = self.remainder_block_length
            index = index - self.last_idx_in_full_block

        block_idx = index // block_length
        dataset_idx = block_idx % len(self.datasets)
        offset = index % block_length

        item = self.datasets[dataset_idx][block_idx * block_length + offset]
        return item

    def __len__(self) -> int:
        return len(self.datasets[0]) * len(self.datasets)  # type: ignore
