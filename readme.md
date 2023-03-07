# GPU-utils

Подразумевается, что эта тулза используется для распределения нагрузки между видеокартами
Она предоставляет класс `GPU` который содержит базовую информацию о видеокарте

Теоретически, можно использовать несколько способов, чтобы доставать информацию о видеокартах, например через

- nvidia-smi
- torch
- напрямую обратиться к ОС

Эта тулза не предоставляет ограничений по способу добычи информации и легко расширяется

## Пример использования

### Установка
```python
from gpu_utilities.gpu_info_obtainer import TorchGPUInfoObtainer
from gpu_utilities.gpu_selector import GPUSelectorFactory, GPUSelectStrategy
import torch

if __name__ == "__main__":
    gpu_info_obtainer = TorchGPUInfoObtainer()  # Выбираем использовать PyTorch как источник данных о картах
    gpus_available = gpu_info_obtainer.gpus_available()  # Проверяем, доступны ли данные о картах (cuda_is_available)

    if not gpus_available:
        raise Exception('GPUs are unavailable')

    selector = GPUSelectorFactory().selector(GPUSelectStrategy.BEST_FIT)  # Создаем селектор, который придерживается стратегии BEST_FIT

    gpu = selector.select(gpu_info_obtainer.gpus(), 800 * 1024 * 1024)  # Выбираем карту из возможных, чтобы на ней поместилось 800 MB, согласно стратегии
    print(f"gpu {gpu} is the best option now!")

    torch.cuda.device(gpu.to_torch_device)  # Пример последующей установки в качестве текущего устройства CUDA
```

### Доступные стратегии
Обеспечить возможность использовать gpu_utils как pip пакет
Библиотека реализует три стратегии:

- BEST_FIT (пытается загрузить самую загруженную карту, на которую вмещается модель)
- WORST_FIT (пытается загрузить самую свободную карту, на которую помещается модель)
- RANDOM (Выбирает случайную карту, на которую помещается модель)

Все их можно использовать через `GPUSelectorFactory`
Enum `GPUSelectStrategy` содержит все три стратегии

> Название стратегий взято из управления памятью в ОС и распределению pages. Там BEST_FIT и WORST_FIT относятся к конкретной странице памяти, поэтому там названия отражают wasted memory -- BEST_FIT -- BEST, потому что максимально избегает "закупоривания памяти" на странице
