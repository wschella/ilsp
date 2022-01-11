from typing import *


class Lambda():
    func: Callable[[Any], Any]

    def __init__(self, func: Callable[[Any], Any]) -> None:
        self.func = func  # type: ignore

    def __call__(self, entry: Any) -> Any:
        return self.func(entry)  # type: ignore
