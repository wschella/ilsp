from typing import *
from pathlib import Path

import click

from assessors.application import cli
from assessors.core import Dataset


@cli.command('debug-list-dataset')
@click.argument('path', type=click.Path(exists=True, path_type=Path))
@click.option('-f', '--field', default="inst_index", help="Field to list unique values for")
@click.option('-u', '--unique', is_flag=True, help="List unique values only", default=False)
@click.option('-h', '--head', default=0, help="Number of entries to show")
def list_dataset(path: Path, field: str, unique: bool, head: int) -> None:
    raise NotImplementedError()
    ds = None
    # ds: Dataset = CustomDatasetDescription(path).load_all()
    ds = ds.map(lambda x: x[field])

    if unique:
        ds = ds.unique()

    if head != 0:
        ds = ds.take(head)

    for value in ds:
        print(value)
