from typing import Any

from kedro.io import AbstractDataset, DatasetError


class NoTracerDataset(AbstractDataset[None, None]):
    """Stands in for a tracer when no tracing backend is configured.

    Loads ``None``, which nodes treat as "no callbacks".
    """

    def load(self) -> None:
        return None

    def save(self, data: Any) -> None:
        raise DatasetError(f"{self.__class__.__name__} is read-only.")

    def _describe(self) -> dict[str, Any]:
        return {}
