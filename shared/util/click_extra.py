from dataclasses import fields
from typing import Any, Dict, List

import click
from click_option_group import optgroup
from tensorflow.python.keras.utils.generic_utils import default


def options_from_dataclass(Dataclass, prefix, with_optgroup=False, exclude: List[str] = []):
    """
    Add CLI options for each field of a Python dataclass to a Click command.
    """
    def decorator(f):
        for field in fields(Dataclass):
            if field.name in exclude:
                continue

            name = f"--{prefix}{field.name}"
            if with_optgroup:
                optgroup.option(name, type=field.type)(f)
            else:
                click.option(name)(f)
        return f
    return decorator


def split_arguments(args: Dict[str, Any], prefixes: List[str]) -> List[Dict[str, Any]]:
    """
    Split a dictionary of CLI arguments based on the OptionGroup they belong to.
    OptionGroups are decided based on prefix.
    """
    groups: Dict[str, Dict[str, Any]] = {prefix: {} for prefix in prefixes + ['default']}
    for k, v in args.items():
        group = 'default'
        name = k
        for prefix in prefixes:
            if k.startswith(prefix):
                group = prefix
                name = k.removeprefix(prefix)
                break
        groups[group][name] = v

    return [groups['default']] + [groups[prefix] for prefix in prefixes]
