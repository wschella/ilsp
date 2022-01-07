from typing import *
import pathlib as pathlib
import dataclasses


import click

from assessors.packages.click_dataclass.docstring import get_attribute_docstring

F = TypeVar("F", bound=Callable[..., Any])
FC = TypeVar("FC", Callable[..., Any], click.Command)

# Notes
# - How to make a distinction between an argument and a required option?
#   - We could tag it in the decorator?
# To Do:
# - [ ] Allow variadic arguments, by handling list type seperately
# - [ ] Special handling of Path types


def arguments(dataclass, positional: List[str] = [], exclude: List[str] = []) -> Callable[[FC], FC]:
    """
    Add all the fields of a dataclass to a Click Command.

    Parameters
    ----------
    dataclass : dataclass
        The dataclass to add the fields of.

    positional : Optional[List[str]]
        The names of the fields to be treated as positional arguments, i.e.
        those that are decorated with `@click.argument`.

    exclude : Optional[List[str]]
        The names of the fields to be excluded from the command.

    Returns
    -------
    Callable[[FC], FC]
        A decorator that adds the fields of the dataclass to a Click Command.
    """

    # References
    # https://docs.python.org/3/library/dataclasses.html#dataclasses.fields
    # https://docs.python.org/3/library/dataclasses.html#dataclasses.Field

    def decorator(f: FC) -> FC:
        for field in dataclasses.fields(dataclass):
            if field.name in exclude:
                continue

            if field.name in positional:
                f = field_argument_decorator(field, dataclass)(f)
            else:
                f = field_option_decorator(field, dataclass)(f)

        return f

    return decorator


def field_argument_decorator(field: dataclasses.Field, dataclass) -> Callable[[FC], FC]:
    """
    Create a decorator that adds a click argument to a click command.
    """
    type = extract_click_type(field.type)

    extra_kwargs = {}
    if field.metadata and field.metadata.get('click'):
        extra_kwargs = field.metadata['click']

    return click.argument(field.name, type=type, **extra_kwargs)


def field_option_decorator(field: dataclasses.Field, dataclass) -> Callable[[FC], FC]:
    """
    Create a decorator that adds a click option to a click command.
    """

    docstring = get_attribute_docstring(dataclass, field.name).aggregate()
    help = docstring
    name = f"--{field.name}"

    type = extract_click_type(field.type)

    required = True
    default = None

    if is_present(field.default):
        required = False
        default = field.default

    # default_factory takes precedence over default
    if is_present(field.default_factory):  # type: ignore
        required = False
        default = getattr(field, 'default_factory')()

    extra_kwargs = {}
    if field.metadata and field.metadata.get('click'):
        extra_kwargs = field.metadata['click']

    return click.option(
        name,
        type=type,
        required=required,
        default=default,
        show_default=True,
        help=help,
        **extra_kwargs
    )


SPECIAL_TYPES = [pathlib.Path]


def extract_click_type(field_type):
    field_type = field_type

    if is_optional(field_type):
        field_type = field_type.__args__[0]

    if field_type in SPECIAL_TYPES:
        field_type = handle_special_type(field_type)

    return field_type


def handle_special_type(type):
    if type == pathlib.Path:
        return click.Path(path_type=pathlib.Path)
    else:
        raise ValueError(f"Unknown special type: {type}")


def is_present(value):
    return value is not dataclasses.MISSING


def is_optional(field_type):
    return get_origin(field_type) is Union and \
        type(None) in get_args(field_type)
