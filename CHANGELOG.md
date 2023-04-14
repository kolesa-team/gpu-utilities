# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning of this project doesn't really give any promises at the moment, it has no backward compatibility.
It might change in the future with the release of 1.0.0 version. 


## [0.2.0] -- 2023-04

### Added

- `NvmlGPUInfoObtainer`
    - Implementation of `GPUInfoObtainer`
    - Uses underlying `NVML` and `nvidia-smi` to access GPU Info
- [Bytes conversion utilities](./gpu_utilities/utils/bytes.py)
    - Allows convert different file sizes to each other
    - Uses `KiB`, `MiB`, etc. as size units


### Changed 

- `TorchGPUInfoObtainer` moved to separate module

### Deprecated 

- `TorchGPUInfoObtainer` due to memory allocation issue

