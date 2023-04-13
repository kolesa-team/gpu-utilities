from __future__ import annotations
from enum import Enum, auto


class UnitSize(Enum):
    B = {
        'name': "byte",
        'multiplier': 2**0
    }

    KiB = {
        'name': "kibibyte",
        'multiplier': 2**10,
    }

    MiB = {
        'name': 'mebibyte',
        'multiplier': 2**20,
    }

    GiB = {
        'name': 'gibibyte',
        'multiplier': 2**30,
    }

    TiB = {
        'name': 'tebibyte',
        'multiplier': 2**40,
    }

    PiB = {
        'name': 'pebibyte',
        'multiplier': 2**50,
    }

    def from_bytes(self, bytes) -> int:
        return bytes / self.multiplier()

    def multiplier(self) -> int:
        return self.value['multiplier']

    def traditional_name(self) -> str:
        return self.value['name']


def convert(size: int, from_unit: UnitSize, target_unit: UnitSize):
    if size == 0:
        return 0

    to_bytes = size * from_unit.multiplier()
    return target_unit.from_bytes(to_bytes)

