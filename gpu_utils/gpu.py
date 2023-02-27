class GPU:
    def __init__(
        self,
        name: str,
        max_available_memory_bytes: int,
        current_memory_utlization_bytes: int,
    ) -> None:
        self.name = name
        self.max_available_memory_bytes = max_available_memory_bytes
        self.current_memory_utlization_bytes = current_memory_utlization_bytes

    def __repr__(self) -> str:
        return (
            f"GPU name={self.name} max_available_memory_bytes={self.max_available_memory_bytes} current_memory_utlization_bytes={self.current_memory_utlization_bytes}"
        )
