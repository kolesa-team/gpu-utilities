# pyright: reportGeneralTypeIssues=false, reportOptionalMemberAccess=false
from typing import Any, Dict, List

from pynvml import nvml
from gpu_utilities.gpu import GPU
from gpu_utilities.gpu_info_obtainer import GPUInfoObtainer
from pynvml.smi import nvidia_smi

DEFAULT_UNIT = "MiB"

class NvmlGPUInfoObtainer(GPUInfoObtainer):
    def __init__(self, nvsmi) -> None:
        if nvsmi is None:
            try:
                nvsmi = nvidia_smi.getInstance()
            except nvml.NVMLError:
                nvsmi = None
        self.nvsmi = nvsmi

    def gpus_available(self) -> bool:
        return self.nvsmi is not None

    def gpus(self) -> List[GPU]:
        if not self.gpus_available():
            return []
        nvml_gpus_wrapped = self.__query_gpus()



    def __query_gpus(self) -> Dict[str, Any]:
        gpu_info = self.nvsmi.DeviceQuery('name,uuid,index,memory.free,memory.total,memory.reserved')
        return gpu_info

    def __dict_to_gpu(self, gpu_dict: Dict[str, Any]) -> List[GPU]:
        gpus = []
        for gpu in gpu_dict:
            frame_buffer_memory_usage = gpu["fb_memory_usage"]
            memory_utilization = frame_buffer_memory_usage["total"] - frame_buffer_memory_usage["free"]
            element = GPU(
                    cuda_id=int(gpu["minor_number"]),
                    name=gpu["product_name"],
                    uuid=gpu["uuid"])
            gpus.append(element)

        return gpus
