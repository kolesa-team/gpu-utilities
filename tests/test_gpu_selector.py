import pytest
from gpu_utilities.errors import EmptyGPUListError, NoGPUAvailableError
from gpu_utilities.gpu import GPU
from gpu_utilities.gpu_selector import (
    BestFitGPUSelector,
    GPUSelectorFactory,
    RandomGPUSelector,
    WorstFitGPUSelector,
    GPUSelectStrategy,
)


class TestBestFitGPUSelector:
    def test_select_valid(self):
        gpu_selector = BestFitGPUSelector()
        gpus = [
            GPU(
                0,
                "Geforce 3060",
                max_available_memory_bytes=12000,
                current_memory_utlization_bytes=9000,
            ),
            GPU(
                1,
                "Geforce 3060",
                max_available_memory_bytes=12000,
                current_memory_utlization_bytes=11000,
            ),
            GPU(
                2,
                "Tesla V-100",
                max_available_memory_bytes=32000,
                current_memory_utlization_bytes=12000,
            ),
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
        with pytest.raises(EmptyGPUListError) as exc_info:
            gpu_selector.select(gpus, 2000)
            assert exc_info.value == "Empty list of gpus provided"

    def test_select_no_available_gpu(self):
        gpu_selector = BestFitGPUSelector()
        gpus = [
            GPU(
                0,
                "Geforce 3060",
                max_available_memory_bytes=12000,
                current_memory_utlization_bytes=9000,
            ),
            GPU(
                1,
                "Geforce 3060",
                max_available_memory_bytes=12000,
                current_memory_utlization_bytes=11000,
            ),
        ]

        with pytest.raises(NoGPUAvailableError) as exc_info:
            gpu_selector.select(gpus, 4000)
            assert exc_info.value == "No available GPU found"


class TestWorstFitGPUSelector:
    def test_select_valid(self):
        gpu_selector = WorstFitGPUSelector()
        gpus = [
            GPU(
                0,
                "Geforce 3060",
                max_available_memory_bytes=12000,
                current_memory_utlization_bytes=9000,
            ),
            GPU(
                1,
                "Geforce 3060",
                max_available_memory_bytes=12000,
                current_memory_utlization_bytes=11000,
            ),
            GPU(
                2,
                "Tesla V-100",
                max_available_memory_bytes=32000,
                current_memory_utlization_bytes=12000,
            ),
        ]
        available = gpu_selector.select(gpus, 2000)
        assert available is not None
        assert available.cuda_id == 2
        assert available.name == "Tesla V-100"
        assert available.max_available_memory_bytes == 32000
        assert available.current_memory_utlization_bytes == 12000

    def test_select_empty_gpus_list(self):
        gpu_selector = WorstFitGPUSelector()
        gpus = []
        with pytest.raises(EmptyGPUListError) as exc_info:
            gpu_selector.select(gpus, 2000)
            assert exc_info.value == "Empty list of gpus provided"

    def test_select_no_available_gpu(self):
        gpu_selector = WorstFitGPUSelector()
        gpus = [
            GPU(
                0,
                "Geforce 3060",
                max_available_memory_bytes=12000,
                current_memory_utlization_bytes=9000,
            ),
            GPU(
                1,
                "Geforce 3060",
                max_available_memory_bytes=12000,
                current_memory_utlization_bytes=11000,
            ),
        ]
        with pytest.raises(NoGPUAvailableError) as exc_info:
            gpu_selector.select(gpus, 4000)
            assert exc_info.value == "No available GPU found"


class TestRandomGPUSelector:
    def test_select_valid(self):
        gpu_selector = RandomGPUSelector()
        gpu_1 = GPU(
            0,
            "Geforce 3060",
            max_available_memory_bytes=12000,
            current_memory_utlization_bytes=9000,
        )
        gpu_2 = GPU(
            1,
            "Geforce 3060",
            max_available_memory_bytes=12000,
            current_memory_utlization_bytes=11000,
        )
        gpu_3 = GPU(
            2,
            "Tesla V-100",
            max_available_memory_bytes=32000,
            current_memory_utlization_bytes=12000,
        )

        gpus = [gpu_1, gpu_2, gpu_3]

        for i in range(0, 100):
            available = gpu_selector.select(gpus, 2000)
            assert (
                available == gpu_1 or available == gpu_2 or available == gpu_3
            )

    def test_select_empty_gpus_list(self):
        gpu_selector = RandomGPUSelector()
        gpus = []
        with pytest.raises(EmptyGPUListError) as exc_info:
            gpu_selector.select(gpus, 2000)
            assert exc_info.value == "Empty list of gpus provided"

    def test_select_no_available_gpu(self):
        gpu_selector = WorstFitGPUSelector()
        gpus = [
            GPU(
                0,
                "Geforce 3060",
                max_available_memory_bytes=12000,
                current_memory_utlization_bytes=9000,
            ),
            GPU(
                1,
                "Geforce 3060",
                max_available_memory_bytes=12000,
                current_memory_utlization_bytes=11000,
            ),
        ]
        with pytest.raises(NoGPUAvailableError) as exc_info:
            gpu_selector.select(gpus, 4000)
            assert exc_info.value == "No available GPU found"


class TestGPUSelectorFactory:
    def test_selector(self):
        factory = GPUSelectorFactory()
        assert (
            isinstance(
                factory.selector(GPUSelectStrategy.BEST_FIT),
                BestFitGPUSelector,
            )
            is True
        )
        assert (
            isinstance(
                factory.selector(GPUSelectStrategy.WORST_FIT),
                WorstFitGPUSelector,
            )
            is True
        )
        assert (
            isinstance(
                factory.selector(GPUSelectStrategy.RANDOM), RandomGPUSelector
            )
            is True
        )
