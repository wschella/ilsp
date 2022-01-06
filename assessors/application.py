from typing import *
import logging
from dataclasses import dataclass

import click

import assessors.utils.setup as setup
from assessors.utils.cli import CommandArguments


@dataclass
class CLIArgs(CommandArguments):
    seed: int = 1234


@click.group()
@click.option('--seed', default=1234, help="Seed for random number generators.")
def cli(**kwargs):
    args = CLIArgs(**kwargs)

    setup.setup_logging()
    setup.setup_rng(args.seed)
    logging.info("Successfully started the assessors experimentation CLI.")
