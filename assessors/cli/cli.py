from typing import *
from dataclasses import dataclass

import click

from assessors.cli.shared import CommandArguments


@dataclass
class CLIArgs(CommandArguments):
    def validate(self):
        pass


@click.group()
@click.pass_context
def cli(ctx, **kwargs):
    """
    The root command. Does nothing by itself except collect global arguments (currently none).
    Invoke subcommands instead.
    """
    ctx.obj = CLIArgs(**kwargs).validated()
