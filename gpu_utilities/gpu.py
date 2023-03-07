class GPU:
    def __init__(
        self,
        cuda_id: int,
        name: str,
        max_available_memory_bytes: int,
        current_memory_utlization_bytes: int,
    ) -> None:
        self.cuda_id = cuda_id
        self.name = name
        self.max_available_memory_bytes = max_available_memory_bytes
        self.current_memory_utlization_bytes = current_memory_utlization_bytes

    def __repr__(self) -> str:
        return str(self.__dict__)

    def to_torch_device(self) -> str:
        return f"cuda:{self.cuda_id}"

    def free_memory_bytes(self) -> int:
        return (
            self.max_available_memory_bytes
            - self.current_memory_utlization_bytes
        )
