import click

import pp.utils.setup


@click.group()
@click.option("--seed", type=int, default=1234)
def cli(seed: int):
    pp.utils.setup.setup_logging()
    pp.utils.setup.setup_rng(seed)
