from typing import *

from torch.utils.data import Dataset

from ._shared import T_co, out_of_bounds


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
    remainder_start: int
    remainder_start_ds: int
    remainder_block_length: int

    def __init__(
        self,
        datasets: Sequence[Dataset[T_co]],
        block_length: int = 1,
    ) -> None:
        assert all(len(d) == len(datasets[0]) for d in datasets)  # type: ignore
        self.datasets = datasets
        self.block_length = block_length

        # Precalculate some values
        n_datasets = len(self.datasets)
        ds_length = len(self.datasets[0])  # type: ignore
        ds_full_blocks = ds_length // block_length
        total_full_blocks = ds_full_blocks * n_datasets
        self.remainder_start = total_full_blocks * block_length
        self.remainder_start_ds = ds_full_blocks * block_length

        remainder_size = len(self) - self.remainder_start
        remainder_block_length = remainder_size // n_datasets
        self.remainder_block_length = remainder_block_length

        super().__init__()

    def __getitem__(self, index) -> T_co:
        if index >= len(self):
            raise out_of_bounds(index, self)

        if index < 0:
            index += len(self)

        # The index is in one of the full blocks
        if index < self.remainder_start:
            block_length = self.block_length
            offset = 0
        # The index is in a remainder block
        else:
            block_length = self.remainder_block_length
            offset = self.remainder_start_ds
            index = index - self.remainder_start

        block_idx = index // block_length
        block_idx_ds = block_idx // len(self.datasets)
        dataset_idx = block_idx % len(self.datasets)
        idx_in_block = index % block_length

        item = self.datasets[dataset_idx][
            offset
            + block_idx_ds * block_length
            + idx_in_block
        ]
        return item

    def __len__(self) -> int:
        return len(self.datasets[0]) * len(self.datasets)  # type: ignore
