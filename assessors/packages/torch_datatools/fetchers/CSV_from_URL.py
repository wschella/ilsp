from typing import *
from pathlib import Path
import requests  # type: ignore

from assessors.packages.torch_datatools.sources import DataFrameDataset
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


@requires('pandas', _has_pandas)
def CSV_from_URL(
    name: str,
    url: str,
    location: Optional[Path] = None,
    transform: Optional[Callable[..., Any]] = None,
    target_column: Optional[str] = None,
    target_transform: Optional[Callable[..., Any]] = None,
    download: Optional[bool] = True,
) -> DataFrameDataset:
    """
    Parameters
    ----------
    name: str
        Name of the dataset.
    url: str
        URL to fetch the dataset from. Should be a CSV file.
    target_column: str
        Name of the column to use as target.
    location: Optional[Path]
        Location to store the dataset. If not present, ./datasets/NAME/data.csv is used.

    Returns
    -------
    A DataFrameDataset.
    """
    path = location or Path('./datasets/') / name / 'data.csv'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        if download:
            streaming_download(url, path)
        else:
            raise FileNotFoundError(f'{path} not found and download=False.')

    df = pd.read_csv(path)

    if transform:
        df[df.columns.difference(['b'])] = df[
            df.columns.difference(['b'])].apply(transform)

    if target_transform:
        df.loc[:, [target_column]] = df.loc[  # type: ignore
            :, [target_column]].apply(target_transform)

    return DataFrameDataset(df, target_column=target_column)


def streaming_download(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
