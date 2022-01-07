from dataclasses import dataclass
from pathlib import Path

import click

import assessors.packages.click_dataclass.click_dataclass as click_dataclass


@dataclass
class AppArgs():
    path: Path  # Seed for random number generators.


@click.command()
@click_dataclass.arguments(AppArgs, positional=['path'])
def cli(**kwargs):
    args = AppArgs(**kwargs)
    print(args)
