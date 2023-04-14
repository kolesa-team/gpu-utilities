# pyright: reportGeneralTypeIssues=false, reportOptionalMemberAccess=false
from typing import Any, Dict, List

from pynvml import nvml
from gpu_utilities.gpu import GPU
from gpu_utilities.gpu_info_obtainer import GPUInfoObtainer
from pynvml.smi import nvidia_smi

from gpu_utilities.utils.bytes import UnitSize, convert


class NvmlGPUInfoObtainer(GPUInfoObtainer):
    def __init__(self, nvsmi=None) -> None:
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
        gpus_parsed = self.__dict_to_gpu(nvml_gpus_wrapped["gpu"])

        return gpus_parsed

    def __query_gpus(self) -> Dict[str, Any]:
        gpu_info = self.nvsmi.DeviceQuery(
            "name,uuid,index,memory.free,memory.total,memory.reserved"
        )
        return gpu_info

    def __dict_to_gpu(self, gpu_dict: Dict[str, Any]) -> List[GPU]:
        gpus = []
        for gpu in gpu_dict:
            frame_buffer_memory_usage = gpu["fb_memory_usage"]  # type: ignore
            byte_dict = self.__fb_memory_usage_to_bytes(
                frame_buffer_memory_usage  # type: ignore
            )

            element = GPU(
                cuda_id=int(gpu["minor_number"]),  # type: ignore
                name=gpu["product_name"],  # type: ignore
                max_available_memory_bytes=byte_dict["total"],
                current_memory_utlization_bytes=byte_dict["utilization"],
                uuid=gpu["uuid"],  # type: ignore
            )
            gpus.append(element)

        return gpus

    def __fb_memory_usage_to_bytes(
        self, fb_dict: Dict[str, Any]
    ) -> Dict[str, int]:
        memory_utilization = fb_dict["total"] - fb_dict["free"]
        unit = UnitSize[fb_dict["unit"]]
        return {
            "total": convert(fb_dict["total"], unit, UnitSize.B),
            "utilization": convert(memory_utilization, unit, UnitSize.B),
            "free": convert(fb_dict["free"], unit, UnitSize.B),
        }
