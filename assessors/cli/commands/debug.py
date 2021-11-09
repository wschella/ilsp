from typing import *
from pathlib import Path

import click

from assessors.cli.cli import cli
from assessors.core import CustomDataset
import assessors.utils.dataset_extra as dse


@cli.command('debug-peek-dataset')
@click.argument('path', type=click.Path(exists=True))
@click.option('-w', '--without-fields', multiple=True, help="Fields to exclude from the dataset")
def peek_dataset(path: Path, without_fields: List[str]) -> None:
    ds = CustomDataset(path).load_all()

    if without_fields != []:
        def filter_fields(entry):
            for field in without_fields:
                entry.pop(field, None)
            return entry
        ds = ds.map(lambda x: filter_fields(x))

    head = ds.take(1).as_numpy_iterator().next()
    print(head)


@cli.command('debug-list-dataset')
@click.argument('path', type=click.Path(exists=True))
@click.option('-f', '--field', default="inst_index", help="Field to list unique values for")
@click.option('-u', '--unique', is_flag=True, help="List unique values only", default=False)
def list_dataset(path: Path, field: str, unique: bool) -> None:
    ds = CustomDataset(path).load_all()
    ds = ds.map(lambda x: x[field])
    if unique:
        ds = ds.unique()

    for value in ds:
        print(value)


@cli.command('debug-list-dataset-split')
@click.argument('path', type=click.Path(exists=True))
@click.option('-f', '--field', default="inst_index", help="Field to list unique values for")
@click.option('-s', '--split-at', default=-10000, help="Split the list at this value")
@click.option('-u', '--unique', is_flag=True, help="List unique values only", default=False)
def list_dataset_split(path: Path, field: str, split_at: int, unique: bool) -> None:
    ds = CustomDataset(path).load_all()
    ds = ds.map(lambda x: x[field])
    (train, test) = dse.split_absolute(ds, split_at)
    if unique:
        train = train.unique()
        test = test.unique()

    for value in train:
        print(value)

    input("Press enter to continue...")
    for value in test:
        print(value)
