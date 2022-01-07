from typing import *
from dataclasses import dataclass

import click

import assessors.utils.setup as setup
from assessors.utils.cli import CommandArguments
from assessors.packages.click_dataclass import click_dataclass


@dataclass
class CLIArgs(CommandArguments):
    seed: int = 1234  # Seed for random number generators.


@click.group()
@click_dataclass.arguments(CLIArgs)
@click.pass_context
def cli(ctx, **kwargs):
    args = CLIArgs(**kwargs)
    ctx.obj = args

    setup.setup_logging()
    setup.setup_rng(args.seed)
