from abc import ABC, abstractmethod
from typing import List

import torch

from gpu_utilities.gpu import GPU


class GPUInfoObtainer(ABC):
    @abstractmethod
    def gpus_available(self) -> bool:
        pass

    @abstractmethod
    def gpus(self) -> List[GPU]:
        pass


class TorchGPUInfoObtainer(GPUInfoObtainer):
    def gpus_available(self) -> bool:
        return torch.cuda.is_available()

    def gpus(self) -> List[GPU]:
        if not self.gpus_available:  # type: ignore
            return []

        gpu_count = torch.cuda.device_count()
        gpus = []
        for gpu_device in range(gpu_count):
            gpu = self.__gpu_device_to_object(gpu_device)
            gpus.append(gpu)

        return gpus

    def __gpu_device_to_object(self, device_id: int) -> GPU:
        name = torch.cuda.get_device_name(device_id)
        memory_usage = torch.cuda.mem_get_info(device_id)
        gpu = GPU(
            cuda_id=device_id,
            name=name,
            max_available_memory_bytes=memory_usage[1],
            current_memory_utlization_bytes=memory_usage[1] - memory_usage[0],
        )
        return gpu
