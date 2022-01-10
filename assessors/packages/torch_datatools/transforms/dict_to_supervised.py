from typing import *


class DictToSupervised():
    """
    Convert a Dataset with Dict entries to one with Tuple[x, y] entries
    for use with supervised learning problems.
    """

    def __init__(self, x: Any, y: Any) -> None:
        """
        Parameters
        ----------
        x : Any
            The key where the input/features/x is at.
        y : Any
            The key where the target/label/y is at.
        """
        self.x = x
        self.y = y

    def __call__(self, entry: Dict[Any, Any]) -> Any:
        return (entry[self.x], entry[self.y])
