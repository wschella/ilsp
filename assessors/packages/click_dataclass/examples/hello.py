from dataclasses import dataclass

import click

import assessors.packages.click_dataclass.click_dataclass as click_dataclass


@dataclass
class AppArgs():
    seed: int = 1234  # Seed for random number generators.


@click.group()
@click_dataclass.arguments(AppArgs)
def cli(**kwargs):
    args = AppArgs(**kwargs)
    print(args)


@dataclass
class HelloArgs():
    name: str


@cli.command('hello')
@click_dataclass.arguments(HelloArgs, positional=['name'])
def hello(**kwargs):
    args = HelloArgs(**kwargs)
    print(f"Hello {args.name}!")
