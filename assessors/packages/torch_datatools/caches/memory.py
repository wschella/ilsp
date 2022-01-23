from typing import TypeVar, Optional, Dict

from ._cache import Cache

T = TypeVar('T')


class MemoryCache(Cache[T]):
    cache: Dict[int, T] = {}

    def __init__(self) -> None:
        self.cache = {}

    def __contains__(self, index: int) -> bool:
        return index in self.cache

    def __getitem__(self, index: int) -> Optional[T]:
        return self.cache.get(index)

    def __setitem__(self, index: int, value: T) -> None:
        self.cache[index] = value
