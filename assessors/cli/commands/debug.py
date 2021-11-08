from typing import *
from dataclasses import dataclass, field
from pathlib import *

import click

from assessors.cli.shared import CommandArguments
from assessors.cli.cli import cli, CLIArgs
from assessors.core import CustomDataset


@dataclass
class PeekDatasetArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    path: Path = Path("./artifacts/dataset")
    without_fields: List[str] = field(default_factory=list)

    def validate(self):
        self.parent.validate()


@cli.command('peek-dataset')
@click.argument('path', type=click.Path(exists=True))
@click.option('-w', '--without-fields', multiple=True, help="Fields to exclude from the dataset")
@click.pass_context
def peek_dataset(ctx, **kwargs):
    args = PeekDatasetArgs(parent=ctx.obj, **kwargs).validated()
    ds = CustomDataset(args.path).load_all()

    if args.without_fields != []:
        def without_fields(entry):
            for field in args.without_fields:
                entry.pop(field, None)
            return entry
        ds = ds.map(lambda x: without_fields(x))

    head = ds.take(1).as_numpy_iterator().next()
    print(head)
