from typing import *
from dataclasses import dataclass
from pathlib import *

import click

from assessors.cli.cli import cli, CLIArgs
from assessors.cli.shared import CommandArguments, DatasetHub
from assessors.core import DatasetDescription


@dataclass
class DownloadArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    name: str = "mnist"

    def validate(self):
        self.parent.validate()


@cli.command('dataset-download')
@click.argument('name')
@click.pass_context
def dataset_download(ctx, **kwargs):
    """
    Download dataset NAME. 
    This command should generally not be necessary, as it happens automatically as well.
    See the DatasetHub for more information on supported datasets.
    """
    args = DownloadArgs(parent=ctx.obj, **kwargs).validated()
    dataset: DatasetDescription = DatasetHub.get(args.name)
    dataset.download()
