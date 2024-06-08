from typing import Callable, Generic, TypeVar

T = TypeVar('T')


class TransformationPipline(Generic[T]):

    def __init__(self):
        self._transformations: list[Callable[[T], T]] = []
        self._last_transformation = None

    def compose(self, transformation: Callable[[T], T], last: bool = False):
        if last and not self._last_transformation:
            self._last_transformation = transformation
        else:
            self._transformations.append(transformation)

    def __call__(self, obj: T):
        return self.transform(obj)

    def transform(self, obj: T):
        for transformation in self._transformations:
            obj = transformation(obj)
        if self._last_transformation:
            obj = self._last_transformation(obj)
        return obj