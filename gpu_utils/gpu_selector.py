from abc import ABC, abstractmethod
from typing import List, Optional

from pydash import filter_, sort_by

from gpu_utils.gpu import GPU


class GPUSelector(ABC):
    @abstractmethod
    def select(
        self, gpus: List[GPU], expected_memory_consumption_bytes: int
    ) -> Optional[GPU]:
        pass


class BestFitGPUSelector(GPUSelector):
    def __init__(self) -> None:
        super().__init__()

    def select(
        self, gpus: List[GPU], expected_memory_consumption_bytes: int
    ) -> Optional[GPU]:
        if len(gpus) == 0:
            raise Exception("Empty list of gpus provided")

        available = filter_(
            gpus,
            lambda gpu: gpu.free_memory_bytes()
            > expected_memory_consumption_bytes,
        )
        if len(available) == 0:
            return None

        sorted = sort_by(
            available, lambda gpu: gpu.free_memory_bytes(), reverse=False
        )
        return sorted[0]
