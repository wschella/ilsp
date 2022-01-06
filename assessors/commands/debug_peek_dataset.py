from typing import *
from pathlib import Path

import click

from assessors.application import cli


@cli.command('debug-peek-dataset')
@click.argument('path', type=click.Path(exists=True, path_type=Path))
@click.option('-w', '--without-fields', multiple=True, help="Fields to exclude from the dataset")
def peek_dataset(path: Path, without_fields: List[str]) -> None:
    raise NotImplementedError()
    ds = None
    # ds: Dataset = CustomDatasetDescription(path).load_all()

    if without_fields != []:
        def filter_fields(entry):
            for field in without_fields:
                entry.pop(field, None)
            return entry
        ds = ds.map(lambda x: filter_fields(x))

    head = next(ds.take(1).as_numpy_sequence())  # type: ignore
    print(head)
