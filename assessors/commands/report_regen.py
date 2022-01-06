from typing import *
from dataclasses import dataclass
from pathlib import Path

import click

from assessors.utils.cli import CommandArguments
from assessors.application import cli, CLIArgs
from assessors.commands.report import generate_report


@dataclass
class ReportRegenArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    results: Optional[Path] = None

    def validate(self):
        self.parent.validate()


@cli.command(name='report-regen')
@click.option('-r', '--results', default=None, type=click.Path(exists=True, path_type=Path), help="The results file to use")
@click.pass_context
def regen_report(ctx, **kwargs):
    args = ReportRegenArgs(parent=ctx.obj, **kwargs).validated()

    files = [args.results] if args.results else list(
        Path('./artifacts/results/').glob("*/results.csv"))

    for f in files:
        ctx.invoke(
            generate_report,
            results=f,
            output_path=f.with_suffix(".html"),
            overwrite=True)
