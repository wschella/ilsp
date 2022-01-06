from abc import ABC, abstractmethod


class CommandArguments(ABC):
    """
    An abstract class for the various dataclasses representing command arguments to extend.
    """

    def validate(self):
        pass

    def validated(self):
        self.validate()
        return self

    def validate_option(self, arg_name, options):
        value = getattr(self, arg_name)
        if value not in options:
            raise ValueError(f"Unknown {arg_name} {value}. Options are {options}.")
