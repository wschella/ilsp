from typing import *
from pathlib import *
from dataclasses import dataclass

from assessors.application import cli
from assessors.utils.cli import CommandArguments
from assessors.hubs import DatasetHub
from assessors.core import DatasetDescription
from assessors.packages.click_dataclass import click_dataclass


@dataclass
class DownloadArgs(CommandArguments):
    name: str
    path: Optional[Path] = None


@cli.command('download-dataset')
@click_dataclass.arguments(DownloadArgs, positional=['name'])
def dataset_download(**kwargs):
    """
    Download dataset NAME. 
    This command should generally not be necessary, as it happens automatically as well.
    See the DatasetHub for more information on supported datasets.
    """
    args = DownloadArgs(**kwargs).validated()
    dataset: DatasetDescription = DatasetHub.get(args.name)
    dataset.download(args.path)
