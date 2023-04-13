from abc import ABC, abstractmethod
from typing import List

from gpu_utilities.gpu import GPU


class GPUInfoObtainer(ABC):
    @abstractmethod
    def gpus_available(self) -> bool:
        pass

    @abstractmethod
    def gpus(self) -> List[GPU]:
        pass
