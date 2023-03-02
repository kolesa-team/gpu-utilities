class GPUUtilsError(Exception):
    pass


class EmptyGPUListError(GPUUtilsError):
    pass


class NoGPUAvailableError(GPUUtilsError):
    pass
