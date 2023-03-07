import logging
from gpu_utilities.gpu_info_obtainer import TorchGPUInfoObtainer


def test_TorchGPUInfoObtainer_gpus():
    info_obtainer = TorchGPUInfoObtainer()
    gpus = info_obtainer.gpus()
    if info_obtainer.gpus_available():
        logging.info(f"gpus are available, gpus={gpus}")
        assert len(gpus) > 0
    else:
        logging.info(f"gpus are not available, gpus={gpus}")
        assert len(gpus) == 0
