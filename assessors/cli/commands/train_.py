from typing import *
from dataclasses import dataclass


from assessors.core import Restore
from assessors.cli.shared import CommandArguments
from assessors.cli.cli import CLIArgs


@dataclass
class TrainArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    restore: Restore.Options = "off"

    def validate(self):
        self.parent.validate()
        self.validate_option("restore", ["full", "checkpoint", "off"])
