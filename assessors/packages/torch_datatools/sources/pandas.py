from typing import *
from typing import TYPE_CHECKING

from torch.utils.data import Dataset

from assessors.packages.torch_datatools.util.requirement import requires


try:
    import pandas as pd
except ImportError:
    _has_pandas = False
else:
    _has_pandas = True

# https://stackoverflow.com/questions/61384752/how-to-type-hint-with-an-optional-import
if TYPE_CHECKING:
    import pandas as pd


class DataFrameDataset(Dataset):
    """
    Make a Torch Dataset out of a pandas DataFrame.

    TODO: Let the return type be configurable, e.g. torch.Tensor, a dict, ...
    Or maybe this should be done with Transforms instead?
    """
    x: pd.DataFrame
    y: pd.DataFrame
    target_column: Optional[str]

    @requires('pandas', _has_pandas)
    def __init__(self, df: pd.DataFrame, target_column: Optional[str]) -> None:
        """
        Parameters
        ----------
        df: pd.DataFrame
            The source DataFrame.
        target_column: Optional[str]
            The name of the column to use as target. If this is present, __getitem__
            will return a tuple of (x, y), where both are pd.Series.
        """
        self.target_column = target_column
        if target_column is not None:
            self.x = df.drop(target_column, axis=1)
            self.y = df.loc[:, [target_column]]
        else:
            self.x = df
            self.y = pd.DataFrame()

        super().__init__()

    def __getitem__(self, index):
        if self.target_column is not None:
            return self.x.iloc[index], self.y.iloc[index]
        else:
            return self.x.iloc[index]

    def __len__(self):
        return len(self.x)
