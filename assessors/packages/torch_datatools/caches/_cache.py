import abc
from typing import Generic, TypeVar, Optional

T = TypeVar('T')


class Cache(abc.ABC, Generic[T]):

    @abc.abstractmethod
    def __contains__(self, index: int) -> bool:
        pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Optional[T]:
        pass

    @abc.abstractmethod
    def __setitem__(self, index: int, value: T) -> None:
        pass
