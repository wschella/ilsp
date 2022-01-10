from typing import *


class SeqToDict():
    """
    Converts a Sequence into a Dict. Entries and `keys` should both have equal length.
    E.g. convert tuples into dicts with SeqToDict(keys=('image', 'label')).
    """
    keys: Sequence[Any]

    def __init__(self, keys: Sequence[Any]) -> None:
        self.keys = keys

    def __call__(self, entry: Any) -> Dict[Any, Any]:
        assert len(entry) == len(self.keys)
        return {key: value for (key, value) in zip(self.keys, entry)}
