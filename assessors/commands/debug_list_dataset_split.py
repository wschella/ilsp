from typing import *
from pathlib import Path

import click

from assessors.application import cli
from assessors.core import Dataset


@cli.command('debug-list-dataset-split')
@click.argument('path', type=click.Path(exists=True, path_type=Path))
@click.option('-f', '--field', default="inst_index", help="Field to list unique values for")
@click.option('-s', '--split-at', default=-10000, help="Split the list at this value")
@click.option('-u', '--unique', is_flag=True, help="List unique values only", default=False)
@click.option('-h', '--head', default=0, help="Number of entries to show")
def list_dataset_split(path: Path, field: str, split_at: int, unique: bool, head: int) -> None:
    raise NotImplementedError()
    ds = None
    # ds: Dataset = CustomDatasetDescription(path).load_all()
    ds = ds.map(lambda x: x[field])
    (train, test) = ds.split_absolute(split_at)

    if unique:
        train = train.unique()
        test = test.unique()

    if head != 0:
        train = train.take(head)
        test = test.take(head)

    for value in train:
        print(value)

    if head:
        print("... split ...")
    else:
        input("Press enter to continue...")

    for value in test:
        print(value)
