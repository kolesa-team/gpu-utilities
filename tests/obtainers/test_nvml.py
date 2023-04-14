import logging

from gpu_utilities.obtainers.nvml import NvmlGPUInfoObtainer


def test_NvmlGPUInfoObtainer_gpus():
    info_obtainer = NvmlGPUInfoObtainer()
    gpus = info_obtainer.gpus()
    if info_obtainer.gpus_available():
        logging.info(f"gpus are available, gpus={gpus}")
        assert len(gpus) > 0
    else:
        logging.info(f"gpus are not available, gpus={gpus}")
        assert len(gpus) == 0
