from typing import *
from dataclasses import dataclass
from pathlib import Path
import os
import json

import click
import pandas as pd
import numpy as np
import sklearn.metrics as metrics

from assessors.cli.shared import CommandArguments
from assessors.cli.cli import cli, CLIArgs
import assessors.report as rr


@dataclass
class ReportArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    results: Path = Path('./results.csv')
    output_path: Path = Path("./results.html")
    overwrite: bool = True

    def validate(self):
        self.parent.validate()


@cli.command(name='report')
@click.argument('results', required=False, default=Path("./results.csv"), type=click.Path(exists=True, path_type=Path))
@click.option('-o', '--output-path', default=Path("./results.html"), type=click.Path(path_type=Path), help="The output path")
@click.option('--overwrite/--no-overwrite', is_flag=True, help="Overwrite the output file if it exists", default=True)
@click.pass_context
def generate_report(ctx, **kwargs):
    """
    Generate a report from a results file
    """

    # Handle CLI args
    args = ReportArgs(parent=ctx.obj, **kwargs).validated()
    if os.path.exists(args.output_path) and not args.overwrite:
        click.confirm(f"The file {args.output_path} already exists. Overwrite?", abort=True)

    with open(args.results, 'r', newline='') as csvfile:
        # Print some simple results
        df = pd.read_csv(csvfile)
        df = rr.wrap.as_classification_with_binary_reward(df)

        rr.cli_report.print_simple(df)

        # And create a full report
        rr.AssessorReport(df).save(args.output_path)
        print(f"Report saved to {args.output_path}")


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
