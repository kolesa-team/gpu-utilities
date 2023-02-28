import pytest
from gpu_utils.gpu import GPU
from gpu_utils.gpu_selector import BestFitGPUSelector


class TestBestFitGPUSelector:
    def test_select_valid(self):
        gpu_selector = BestFitGPUSelector()
        gpus = [
            GPU(0, "Geforce 3060", 12000, 9000),
            GPU(1, "Geforce 3060", 12000, 11000),
            GPU(2, "Tesla V-100", 32000, 12000),
        ]
        available = gpu_selector.select(gpus, 2000)
        assert available is not None
        assert available.cuda_id == 0
        assert available.name == "Geforce 3060"
        assert available.max_available_memory_bytes == 12000
        assert available.current_memory_utlization_bytes == 9000

    def test_select_empty_gpus_list(self):
        gpu_selector = BestFitGPUSelector()
        gpus = []
        with pytest.raises(Exception) as exc_info:
            gpu_selector.select(gpus, 2000)
            assert exc_info.value == "Empty list of gpus provided"

    def test_select_no_available_gpu(self):
        gpu_selector = BestFitGPUSelector()
        gpus = [
            GPU(0, "Geforce 3060", 12000, 9000),
            GPU(1, "Geforce 3060", 12000, 11000),
        ]
        available = gpu_selector.select(gpus, 4000)
        assert available is None
