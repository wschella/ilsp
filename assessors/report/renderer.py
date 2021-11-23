from abc import ABC, abstractmethod
import traceback
from typing import *
from pathlib import Path


class Component(ABC):
    """
    Minimal composable component rendering system for static pages.
    """

    @abstractmethod
    def render(self) -> str:
        pass

    def _repr_html_(self) -> str:
        try:
            return self.render()
        except Exception as e:
            return GenericError(e).render()

    def __str__(self) -> str:
        return self._repr_html_()

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            f.write(self.render())


class GenericError(Component):
    def __init__(self, e: Exception):
        self.e = e

    def render(self) -> str:

        return f'''
        <div class = "alert alert-danger" >
            <h3> {type(self).__name__} </h3>
            <h4> {type(self.e).__name__} </h4>
            <pre> {self.e} </pre>
            <pre> {traceback.format_exc()} </pre>
        <div>
        '''
