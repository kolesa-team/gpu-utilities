from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional
import random
from gpu_utilities.errors import NoGPUAvailableError, EmptyGPUListError
from gpu_utilities.gpu import GPU


class GPUSelectStrategy(Enum):
    BEST_FIT = (auto(),)
    WORST_FIT = (auto(),)
    RANDOM = auto()


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
    ) -> GPU:
        if len(gpus) == 0:
            raise EmptyGPUListError()

        available = list(
            filter(
                lambda gpu: gpu.free_memory_bytes()
                > expected_memory_consumption_bytes,
                gpus,
            )
        )

        if not available:
            raise NoGPUAvailableError()

        sorted_gpus = sorted(
            available, key=lambda gpu: gpu.free_memory_bytes(), reverse=False
        )
        return sorted_gpus[0]


class WorstFitGPUSelector(GPUSelector):
    def __init__(self) -> None:
        super().__init__()

    def select(
        self, gpus: List[GPU], expected_memory_consumption_bytes: int
    ) -> GPU:
        if len(gpus) == 0:
            raise EmptyGPUListError()

        available = list(
            filter(
                lambda gpu: gpu.free_memory_bytes()
                > expected_memory_consumption_bytes,
                gpus,
            )
        )

        if len(available) == 0:
            raise NoGPUAvailableError()

        sorted_gpus = sorted(
            available, key=lambda gpu: gpu.free_memory_bytes(), reverse=True
        )
        return sorted_gpus[0]


class RandomGPUSelector(GPUSelector):
    def __init__(self) -> None:
        super().__init__()

    def select(
        self, gpus: List[GPU], expected_memory_consumption_bytes: int
    ) -> GPU:
        if len(gpus) == 0:
            raise EmptyGPUListError()

        available = list(
            filter(
                lambda gpu: gpu.free_memory_bytes()
                > expected_memory_consumption_bytes,
                gpus,
            )
        )

        if not available:
            raise NoGPUAvailableError()

        id = random.randrange(0, len(available))  # type: ignore
        return available[id]


class GPUSelectorFactory:
    def selector(self, strategy: GPUSelectStrategy) -> GPUSelector:
        if strategy == GPUSelectStrategy.BEST_FIT:
            return BestFitGPUSelector()
        elif strategy == GPUSelectStrategy.WORST_FIT:
            return WorstFitGPUSelector()
        elif strategy == GPUSelectStrategy.RANDOM:
            return RandomGPUSelector()
        else:
            raise Exception(
                f"cannot initialize selector for strategy {strategy}"
            )
