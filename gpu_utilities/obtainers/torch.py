from typing import List
from deprecated import deprecated
import torch
from gpu_utilities.gpu import GPU

from gpu_utilities.gpu_info_obtainer import GPUInfoObtainer


@deprecated(
    version="0.2.0",
    reason=(
        "It allocates undesirable memory in GPU FrameBuffer. To avoid this"
        " issue, use alternative GPUInfoObtainer implementations"
    ),
)  # noqa
class TorchGPUInfoObtainer(GPUInfoObtainer):
    def gpus_available(self) -> bool:
        result = torch.cuda.is_available()
        torch.cuda.empty_cache()
        return result

    def gpus(self) -> List[GPU]:
        if not self.gpus_available:  # type: ignore
            return []

        gpu_count = torch.cuda.device_count()
        gpus = []
        for gpu_device in range(gpu_count):
            gpu = self.__gpu_device_to_object(gpu_device)
            gpus.append(gpu)

        torch.cuda.empty_cache()
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
