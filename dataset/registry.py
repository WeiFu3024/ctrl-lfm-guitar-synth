from typing import Dict, List, Type, TypeVar

T = TypeVar("T")


class Registry:
    """Name -> class lookup used by the dataset/control-signal/latent-encoder registries."""

    def __init__(self, kind: str):
        self._kind = kind
        self._entries: Dict[str, Type] = {}

    def register(self, name: str):
        def deco(cls: Type[T]) -> Type[T]:
            if name in self._entries:
                raise ValueError(f"{self._kind} '{name}' is already registered")
            self._entries[name] = cls
            return cls

        return deco

    def get(self, name: str) -> Type:
        if name not in self._entries:
            raise KeyError(f"Unknown {self._kind} '{name}'. Available: {list(self._entries)}")
        return self._entries[name]

    def names(self) -> List[str]:
        return list(self._entries)
