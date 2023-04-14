# GPU-utils

This is a small library to use in services, that require information about GPU.
Primary use case in mind for developing this utility library: select GPU based on available memory in GPU-intensive services.

There are several methods to acquire GPU information. We could use, for example: 

- nvml 
- nvidia-smi
- torch
- etc.

This library was developed with extensibility in mind, if we were to change primary method of obtaining GPU information 

## Quickstart 

### Usage example 

```python
from gpu_utilities.obtainers.nvml import NvmlGPUInfoObtainer
from gpu_utils.gpu_selector import GPUSelectorFactory, GPUSelectStrategy
import torch

if __name__ == "__main__":
    gpu_info_obtainer = NvmlGPUInfoObtainer()  # Use NVML and nvidia-smi as GPU information provider
    gpus_available = gpu_info_obtainer.gpus_available()  # Check if information about GPU is available

    if not gpus_available:
        raise Exception('GPUs are unavailable')

    selector = GPUSelectorFactory().selector(GPUSelectStrategy.BEST_FIT)  # Create Selector for strategy BEST_FIT

    gpu = selector.select(gpu_info_obtainer.gpus(), 800 * 1024 * 1024)  # Sekect GPU that is able to allocate 800 MB according to strategy
    print(f"gpu {gpu} is the best option now!")

    torch.cuda.device(gpu.to_torch_device())  # Example illustrating usage in torch

```

### GPU info obtaining 

There are 2 implemented methods at the moment to obtain information about GPU.

- ~~`TorchGPUInfoObtainer`~~
    - via `PyTorch` library
    - Deprecated since 0.2.0, since it allocates memory on GPU to be used
- `NvmlGPUInfoObtainer`
    - Requires [NVML](https://developer.nvidia.com/nvidia-management-library-nvml) to be installed on your machine 
    - Might additionally require [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface)


### Selection Strategies

Three very basic strategies to choose GPU implemented. All strategies filter GPU's that can allocate desirable memory span.

- BEST_FIT (Choose the most loaded GPU (memory-wise))
- WORST_FIT (Choose the least loaded GPU (memory-wise))
- RANDOM

They are all accessible via `GPUSelectorFactory`.
Enum `GPUSelectStrategy` contains all implementations.

> Naming of the strategies can be seen as a bit odd. It was inspired by page allocation strategies in the OS, where "BEST_FIT" represented the most effecient use, hence the name.
