from abc import ABC, abstractmethod
from enum import Enum, auto
from random import random
from typing import List, Optional

from pydash import filter_, sort_by

from gpu_utils.gpu import GPU


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
            raise Exception("Empty list of gpus provided")

        available = filter_(
            gpus,
            lambda gpu: gpu.free_memory_bytes()
            > expected_memory_consumption_bytes,
        )
        if len(available) == 0:
            raise Exception("No available GPU found")

        sorted = sort_by(
            available, lambda gpu: gpu.free_memory_bytes(), reverse=False
        )
        return sorted[0]


class WorstFitGPUSelector(GPUSelector):
    def __init__(self) -> None:
        super().__init__()

    def select(
        self, gpus: List[GPU], expected_memory_consumption_bytes: int
    ) -> GPU:
        if len(gpus) == 0:
            raise Exception("Empty list of gpus provided")

        available = filter_(
            gpus,
            lambda gpu: gpu.free_memory_bytes()
            > expected_memory_consumption_bytes,
        )
        if len(available) == 0:
            raise Exception("No available GPU found")

        sorted = sort_by(
            available, lambda gpu: gpu.free_memory_bytes(), reverse=True
        )
        return sorted[0]


class RandomGPUSelector(GPUSelector):
    def __init__(self) -> None:
        super().__init__()

    def select(
        self, gpus: List[GPU], expected_memory_consumption_bytes: int
    ) -> GPU:
        if len(gpus) == 0:
            raise Exception("Empty list of gpus provided")

        available = filter_(
            gpus,
            lambda gpu: gpu.free_memory_bytes()
            > expected_memory_consumption_bytes,
        )
        if len(available) == 0:
            raise Exception("No available GPU found")

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
