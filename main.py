from gpu_utils.gpu_info_obtainer import TorchGPUInfoObtainer


if __name__ == "__main__":
    obtainer = TorchGPUInfoObtainer()
    gpus = obtainer.gpus()
    print(gpus)
